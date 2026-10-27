import consola from "consola"
import { Buffer } from "node:buffer"

import type {
  ResponsesApiRequest,
  ResponsesContentPart,
  ResponsesInputItem,
} from "~/routes/responses/types"

import { copilotBaseUrl, copilotHeaders } from "~/lib/api-config"
import { wrapCancellableResponse } from "~/lib/cancellable-response"
import { copilotFetch } from "~/lib/copilot-fetch"
import { HTTPError } from "~/lib/error"
import { jsonByteLength, truncateJsonString } from "~/lib/json-size"
import {
  getKnownModelPromptLimit,
  getModelPromptLimit,
} from "~/lib/model-limits"
import { state } from "~/lib/state"
import {
  createEncryptedOutputScope,
  isEncryptedOutputError,
  inspectResponsesStreamFailure,
  isRejectedEncryptedOutput,
  rememberRejectedEncryptedOutputs,
  stripEncryptedOutputParts,
} from "~/services/copilot/encrypted-output-recovery"
import { validateResponsesCapabilities } from "~/services/copilot/responses-capabilities"

/**
 * Bytes per token for Responses payloads, used only when opt-in token-derived
 * trimming is enabled.
 *
 * Measured against multi-turn payloads built from this repository's own source
 * (user messages, function_call items, large function_call_output bodies,
 * assistant replies): 4.4 JSON bytes per content token, 3.7 per token of the
 * serialized JSON itself. The previous value of 3.5 was inherited from
 * context-manager.ts, where it measures *unescaped content length* rather than
 * serialized JSON — copying the constant silently changed its basis and made
 * the derived ceiling ~26% too aggressive, engaging trimming at roughly 77% of
 * a model's real window.
 *
 * 4.3 is the conservative end of what was measured. Neither figure is Copilot's
 * own prompt-token count, which is why this no longer gates the default path.
 */
const CHARS_PER_TOKEN_ESTIMATE = 4.3
const TOKEN_RESERVE = 8_000
const IMAGE_STRIPPED_PLACEHOLDER =
  "[image removed to stay under upstream payload limit]"
const INVALID_OUTPUT_IMAGE_PLACEHOLDER =
  "[image output removed because its URL is not valid for Copilot Responses]"
const INPUT_DROPPED_PLACEHOLDER =
  "[older response input omitted to stay under context limit]"
const INPUT_TRUNCATED_PREFIX =
  "[older response input truncated to stay under context limit]\n\n"
const PRESERVED_INPUT_STRING_KEYS = new Set([
  "call_id",
  "id",
  "model",
  "name",
  "previous_response_id",
  "role",
  "status",
  "tool_call_id",
  "type",
])

export async function createResponses(
  payload: ResponsesApiRequest,
  signal?: AbortSignal,
): Promise<Response> {
  if (!state.copilotToken) throw new Error("Copilot token not found")
  validateResponsesCapabilities(payload)
  const cancellation = new AbortController()
  const upstreamSignal =
    signal ?
      AbortSignal.any([signal, cancellation.signal])
    : cancellation.signal
  upstreamSignal.throwIfAborted()

  const url = `${copilotBaseUrl(state)}/responses`
  const encryptedOutputScope = createEncryptedOutputScope(
    url,
    payload.model,
    state.githubToken ?? state.copilotToken,
  )
  const knownRejected = stripEncryptedOutputParts(
    sanitizeResponsesPayload(payload),
    (fingerprint) =>
      isRejectedEncryptedOutput(encryptedOutputScope, fingerprint),
  )
  const upstreamPayload = fitResponsesPayload(
    knownRejected.payload,
    computeResponsesPayloadCeiling(payload.model),
  )
  const headers: Record<string, string> = {
    ...copilotHeaders(state, responsesPayloadHasImages(upstreamPayload)),
    accept: upstreamPayload.stream ? "text/event-stream" : "application/json",
    "X-Initiator": "agent",
  }
  const bodyLength = jsonByteLength(upstreamPayload)

  consola.debug(
    `Sending responses payload: ${bodyLength} bytes, model: ${payload.model}`,
  )

  const result = await fetchResponsesWithEncryptedOutputFallback(
    { url, headers, encryptedOutputScope, signal: upstreamSignal },
    upstreamPayload,
  )
  const { response } = result

  if (!response.ok) {
    const errorBody = result.errorBody ?? (await response.text())
    const requestBytes = Buffer.byteLength(result.body, "utf8")
    consola.error(
      `Failed to create responses - Status: ${response.status} ${response.statusText}`,
    )
    consola.error(`Response body: ${errorBody}`)
    consola.error(`Request payload size: ${requestBytes} bytes`)

    if (isContextOverflow(response, errorBody, requestBytes)) {
      const modelCaps = state.models?.data.find((m) => m.id === payload.model)
        ?.capabilities.limits
      const maxOutputTokens = payload.max_output_tokens ?? 0

      // Prefer the numbers Copilot reported over a local estimate. Claude Code
      // uses the gap between them to size its compaction pass, so substituting
      // a bytes/4 guess for upstream's real count makes it compact by the wrong
      // amount. Fall back to the estimate only when upstream gave us nothing.
      const upstream = parseUpstreamTokenCounts(errorBody)
      const estimatedTokens = upstream.used ?? Math.ceil(requestBytes / 4)
      const modelLimit =
        upstream.limit ?? getModelPromptLimit(payload.model, modelCaps)
      const source = upstream.used ? "upstream-reported" : "estimated"

      consola.warn(
        `Responses context overflow -> returning 400 prompt-too-long (${source}: ~${estimatedTokens} + ${maxOutputTokens} > ${modelLimit})`,
      )

      throw new HTTPError(
        "Prompt too long",
        new Response(
          JSON.stringify({
            type: "error",
            error: {
              type: "invalid_request_error",
              code: "context_length_exceeded",
              message: `prompt is too long: input length and \`max_tokens\` exceed context limit: ${estimatedTokens} + ${maxOutputTokens} > ${modelLimit} tokens`,
              // Preserve what upstream actually said, so a client that wants
              // to diagnose rather than just compact is not flying blind.
              upstream_error: errorBody.slice(0, 2_000),
            },
          }),
          {
            status: 400,
            statusText: "Bad Request",
            headers: { "content-type": "application/json" },
          },
        ),
      )
    }

    throw new HTTPError(
      "Failed to create responses",
      new Response(errorBody, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      }),
    )
  }

  return wrapCancellableResponse(response, cancellation)
}

interface ResponsesFetchResult {
  body: string
  errorBody?: string
  response: Response
}

interface ResponsesFetchContext {
  signal: AbortSignal
  encryptedOutputScope: string
  headers: Record<string, string>
  url: string
}

async function fetchResponsesWithEncryptedOutputFallback(
  context: ResponsesFetchContext,
  payload: ResponsesApiRequest,
): Promise<ResponsesFetchResult> {
  const { encryptedOutputScope, signal } = context
  const fallback = stripEncryptedOutputParts(payload)
  const initial = await sendResponsesRequest(context, payload)

  if (
    fallback.count > 0
    && !initial.response.ok
    && initial.errorBody
    && isEncryptedOutputError(initial.response, initial.errorBody)
  ) {
    consola.warn(
      `Copilot could not decrypt ${fallback.count} encrypted output part(s); retrying without them`,
    )
    rememberRejectedEncryptedOutputs(
      encryptedOutputScope,
      fallback.fingerprints,
    )
    return fetchResponsesWithEncryptedOutputFallback(context, fallback.payload)
  }

  if (!initial.response.ok || !payload.stream) return initial
  const inspected = await inspectResponsesStreamFailure(
    initial.response,
    signal,
  )
  initial.response = inspected.response
  if (
    inspected.errorBody
    && isContextOverflow(
      new Response(null, { status: 400 }),
      inspected.errorBody,
      0,
    )
  ) {
    void inspected.response.body?.cancel().catch(() => undefined)
    return {
      ...initial,
      errorBody: inspected.errorBody,
      response: new Response(inspected.errorBody, { status: 400 }),
    }
  }
  if (inspected.failed && fallback.count > 0) {
    void inspected.response.body?.cancel().catch(() => undefined)
    consola.warn(
      `Copilot stream failed before output; retrying without ${fallback.count} encrypted output part(s)`,
    )
    rememberRejectedEncryptedOutputs(
      encryptedOutputScope,
      fallback.fingerprints,
    )
    return fetchResponsesWithEncryptedOutputFallback(context, fallback.payload)
  }

  return initial
}

async function sendResponsesRequest(
  context: ResponsesFetchContext,
  payload: ResponsesApiRequest,
): Promise<ResponsesFetchResult> {
  const { url, headers, signal } = context
  const body = JSON.stringify(payload)
  const response = await copilotFetch(url, {
    method: "POST",
    headers,
    body,
    signal,
  })
  const errorBody = response.ok ? undefined : await response.text()
  return { body, errorBody, response }
}

function fitResponsesPayload(
  payload: ResponsesApiRequest,
  ceiling: number,
): ResponsesApiRequest {
  const initialBytes = jsonByteLength(payload)
  if (initialBytes <= ceiling) return payload

  consola.info(
    `Responses context fit: payload ${initialBytes} bytes exceeds ${ceiling} byte ceiling - reducing`,
  )

  if (typeof payload.input === "string") {
    const current = truncateStringInput(payload, ceiling)
    const currentBodyLength = jsonByteLength(current)
    consola.warn(
      `Responses context fit: truncated string input (${initialBytes} -> ${currentBodyLength} bytes)`,
    )
    return current
  }

  let current = payload
  let currentBodyLength = initialBytes

  const imageLocations = collectImageLocations(payload)
  let strippedCount = 0

  for (const location of imageLocations) {
    current = replaceImageWithPlaceholder(current, location)
    strippedCount++
    currentBodyLength = jsonByteLength(current)
    if (currentBodyLength <= ceiling) break
  }

  if (strippedCount > 0) {
    consola.warn(
      `Responses context fit: stripped ${strippedCount} images (${initialBytes} -> ${currentBodyLength} bytes)`,
    )
  }

  if (currentBodyLength <= ceiling) return current

  const dropped = dropOldInputItems(current, ceiling)
  current = dropOrphanedToolCallOutputs(dropped.payload)
  currentBodyLength = jsonByteLength(current)
  if (dropped.count > 0) {
    consola.warn(
      `Responses context fit: dropped ${dropped.count} old input items (${initialBytes} -> ${currentBodyLength} bytes)`,
    )
  }

  if (currentBodyLength <= ceiling) return current

  current = truncateLargestInputContent(current, ceiling)
  currentBodyLength = jsonByteLength(current)
  consola.warn(
    `Responses context fit: truncated input content (${initialBytes} -> ${currentBodyLength} bytes)`,
  )

  if (currentBodyLength <= ceiling) return current

  current = minimizeResponsesInput(current, ceiling)
  currentBodyLength = jsonByteLength(current)
  consola.warn(
    `Responses context fit: minimized input history (${initialBytes} -> ${currentBodyLength} bytes)`,
  )

  if (currentBodyLength <= ceiling) return current

  current = truncateLargestPayloadStrings(current, ceiling)
  currentBodyLength = jsonByteLength(current)
  consola.warn(
    `Responses context fit: truncated payload strings (${initialBytes} -> ${currentBodyLength} bytes)`,
  )

  return current
}

/**
 * Configurable local byte budget, independent of the upstream's current limits.
 * Opt-in token-derived trimming can impose a smaller budget.
 */
function computeResponsesPayloadCeiling(modelId: string): number {
  const maxPayloadBytes = state.responsesMaxPayloadBytes
  if (!state.responsesContextTrim) return maxPayloadBytes

  const limits = state.models?.data.find((m) => m.id === modelId)?.capabilities
    .limits
  const maxPromptTokens = getKnownModelPromptLimit(modelId, limits)
  if (!maxPromptTokens) return maxPayloadBytes

  const tokenDerivedBytes = Math.floor(
    (maxPromptTokens - TOKEN_RESERVE) * CHARS_PER_TOKEN_ESTIMATE,
  )
  return Math.min(maxPayloadBytes, tokenDerivedBytes)
}

function truncateStringInput(
  payload: ResponsesApiRequest,
  ceiling: number,
): ResponsesApiRequest {
  if (typeof payload.input !== "string") return payload

  const overhead =
    jsonByteLength({ ...payload, input: "" }) - jsonByteLength("")
  return {
    ...payload,
    input: truncateJsonString(
      payload.input,
      ceiling - overhead,
      INPUT_TRUNCATED_PREFIX,
    ),
  }
}

interface ImageLocation {
  contentIndex: number
  inputIndex: number
}

function collectImageLocations(
  payload: ResponsesApiRequest,
): Array<ImageLocation> {
  if (typeof payload.input === "string") return []

  const locations: Array<ImageLocation> = []
  for (const [inputIndex, item] of payload.input.entries()) {
    if (!Array.isArray(item.content)) continue

    for (const [contentIndex, part] of item.content.entries()) {
      if (part.type === "input_image") {
        locations.push({ contentIndex, inputIndex })
      }
    }
  }
  return locations
}

function replaceImageWithPlaceholder(
  payload: ResponsesApiRequest,
  location: ImageLocation,
): ResponsesApiRequest {
  if (typeof payload.input === "string") return payload

  const input = [...payload.input]
  const item = input[location.inputIndex]
  if (!Array.isArray(item.content)) return payload

  const content = [...item.content]
  content[location.contentIndex] = {
    type: "input_text",
    text: IMAGE_STRIPPED_PLACEHOLDER,
  }

  input[location.inputIndex] = { ...item, content }
  return { ...payload, input }
}

/**
 * Replace the oldest history with placeholders until the payload fits.
 *
 * Walks oldest → newest: the trailing items carry the active request, so
 * anything dropped has to come off the front. This is unavoidably hostile to
 * prompt caching — a cache is keyed on a stable prefix, and rewriting item 0
 * invalidates everything after it — but the alternative (dropping recent turns
 * to preserve the cacheable prefix) throws away exactly the context the model
 * needs most. The real mitigation is not running this at all by default; see
 * computeResponsesPayloadCeiling.
 *
 * The two most recent droppable items are always preserved, as is anything
 * system/developer.
 */
function dropOldInputItems(
  payload: ResponsesApiRequest,
  ceiling: number,
): { bodyLength: number; count: number; payload: ResponsesApiRequest } {
  if (typeof payload.input === "string") {
    return { bodyLength: jsonByteLength(payload), count: 0, payload }
  }

  const input = [...payload.input]
  let bodyLength = jsonByteLength({ ...payload, input })
  let count = 0

  for (let index = 0; index < input.length && bodyLength > ceiling; index++) {
    const item = input[index]
    if (item.role === "system" || item.role === "developer") continue
    if (countRecentDroppableItems(input, index) <= 2) continue

    input[index] = createDroppedInputItem(item)
    count++
    bodyLength = jsonByteLength({ ...payload, input })
  }

  return { bodyLength, count, payload: { ...payload, input } }
}

function createDroppedInputItem(item: ResponsesInputItem): ResponsesInputItem {
  return {
    role: item.role === "assistant" ? "assistant" : "user",
    content: INPUT_DROPPED_PLACEHOLDER,
  }
}

function dropOrphanedToolCallOutputs(
  payload: ResponsesApiRequest,
): ResponsesApiRequest {
  if (typeof payload.input === "string") return payload

  const callIds = new Set(
    payload.input
      .filter((item) => isToolCallItem(item) && item.call_id)
      .map((item) => item.call_id),
  )
  let droppedCount = 0
  const input = payload.input.map((item) => {
    if (
      !isToolCallOutputItem(item)
      || !item.call_id
      || callIds.has(item.call_id)
    ) {
      return item
    }

    droppedCount++
    return createDroppedInputItem(item)
  })

  if (droppedCount === 0) return payload
  consola.warn(
    `Responses context fit: dropped ${droppedCount} orphaned tool call outputs`,
  )
  return { ...payload, input }
}

function isToolCallItem(item: ResponsesInputItem): boolean {
  return typeof item.type === "string" && item.type.endsWith("_call")
}

function isToolCallOutputItem(item: ResponsesInputItem): boolean {
  return typeof item.type === "string" && item.type.endsWith("_call_output")
}

function countRecentDroppableItems(
  input: Array<ResponsesInputItem>,
  index: number,
): number {
  let count = 0
  for (let i = index; i < input.length; i++) {
    if (input[i].role !== "system" && input[i].role !== "developer") count++
  }
  return count
}

function truncateLargestInputContent(
  payload: ResponsesApiRequest,
  ceiling: number,
): ResponsesApiRequest {
  if (typeof payload.input === "string") {
    return truncateStringInput(payload, ceiling)
  }

  let current = payload
  let bodyLength = jsonByteLength(current)

  while (bodyLength > ceiling) {
    const target = findLargestInputString(current)
    if (!target) return current

    current = truncateStringAtLocation(current, target, bodyLength - ceiling)
    const nextBodyLength = jsonByteLength(current)
    if (nextBodyLength >= bodyLength) return current
    bodyLength = nextBodyLength
  }

  return current
}

interface StringLocation extends StringCandidate {
  inputIndex: number
}

function findLargestInputString(
  payload: ResponsesApiRequest,
): StringLocation | null {
  if (typeof payload.input === "string") return null

  let largest: StringLocation | null = null
  for (const [inputIndex, item] of payload.input.entries()) {
    if (item.role === "system" || item.role === "developer") continue

    const candidate = findLargestStringInValue(item, [])
    if (candidate && (!largest || candidate.length > largest.length)) {
      largest = { inputIndex, ...candidate }
    }
  }

  return largest
}

function truncateStringAtLocation(
  payload: ResponsesApiRequest,
  location: StringLocation,
  excessBytes: number,
): ResponsesApiRequest {
  if (typeof payload.input === "string") return payload

  const input = [...payload.input]
  input[location.inputIndex] = truncateStringAtPath(
    input[location.inputIndex],
    location.path,
    excessBytes,
  ) as ResponsesInputItem
  return { ...payload, input }
}

interface StringCandidate {
  length: number
  path: Array<number | string>
}

function findLargestStringInValue(
  value: unknown,
  path: Array<number | string>,
): StringCandidate | null {
  if (typeof value === "string") {
    const key = path.at(-1)
    if (typeof key === "string" && PRESERVED_INPUT_STRING_KEYS.has(key)) {
      return null
    }
    return { length: value.length, path }
  }

  if (Array.isArray(value)) {
    return value.reduce<StringCandidate | null>((largest, item, index) => {
      const candidate = findLargestStringInValue(item, [...path, index])
      if (!candidate) return largest
      return !largest || candidate.length > largest.length ? candidate : largest
    }, null)
  }

  if (!isRecord(value)) return null

  let largest: StringCandidate | null = null
  for (const [key, nested] of Object.entries(value)) {
    const candidate = findLargestStringInValue(nested, [...path, key])
    if (candidate && (!largest || candidate.length > largest.length)) {
      largest = candidate
    }
  }
  return largest
}

function truncateStringAtPath(
  value: unknown,
  path: Array<number | string>,
  excessBytes: number,
): unknown {
  if (path.length === 0) {
    if (typeof value !== "string") return value
    return truncateJsonString(
      value,
      jsonByteLength(value) - excessBytes,
      INPUT_TRUNCATED_PREFIX,
    )
  }

  const [head, ...tail] = path
  if (Array.isArray(value) && typeof head === "number") {
    const arrayValue = value as Array<unknown>
    const next = [...arrayValue]
    next[head] = truncateStringAtPath(next[head], tail, excessBytes)
    return next
  }

  if (isRecord(value) && typeof head === "string") {
    return {
      ...value,
      [head]: truncateStringAtPath(value[head], tail, excessBytes),
    }
  }

  return value
}

function minimizeResponsesInput(
  payload: ResponsesApiRequest,
  ceiling: number,
): ResponsesApiRequest {
  if (typeof payload.input === "string") {
    return truncateStringInput(payload, ceiling)
  }

  const protectedItems = payload.input.filter(
    (item) => item.role === "system" || item.role === "developer",
  )
  const latest = [...payload.input]
    .reverse()
    .find((item) => item.role !== "system" && item.role !== "developer")

  return {
    ...payload,
    input: [
      ...protectedItems,
      {
        role: latest?.role === "assistant" ? "assistant" : "user",
        content: INPUT_DROPPED_PLACEHOLDER,
      },
    ],
  }
}

function truncateLargestPayloadStrings(
  payload: ResponsesApiRequest,
  ceiling: number,
): ResponsesApiRequest {
  let current = payload
  let bodyLength = jsonByteLength(current)

  while (bodyLength > ceiling) {
    const target = findLargestStringInValue(current, [])
    if (!target) return current

    current = truncateStringAtPath(
      current,
      target.path,
      bodyLength - ceiling,
    ) as ResponsesApiRequest
    const nextBodyLength = jsonByteLength(current)
    if (nextBodyLength >= bodyLength) return current
    bodyLength = nextBodyLength
  }

  return current
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

/**
 * Error markers that unambiguously mean "the prompt did not fit". These are
 * upstream saying so explicitly, so they are safe to act on regardless of
 * payload size.
 */
const CONTEXT_OVERFLOW_PATTERNS = [
  /request entity too large/i,
  /exceeds the limit of \d+/i,
  /context_length_exceeded/i,
  /model_max_prompt_tokens_exceeded/i,
  /payload too large/i,
  /maximum context length/i,
  /prompt is too long/i,
]

/**
 * Markers of a transient upstream outage. These can coincide with a large
 * payload without the payload being the cause, so they must never be read as
 * context overflow — doing so makes a client compact away history to work
 * around what is really a Copilot hiccup.
 */
const TRANSIENT_OUTAGE_PATTERNS = [
  /service unavailable/i,
  /bad gateway/i,
  /temporarily unavailable/i,
  /upstream connect error/i,
  /overloaded/i,
  /try again later/i,
]

function isContextOverflow(
  response: Response,
  errorBody: string,
  bodyLength: number,
): boolean {
  if (response.status === 413) return true
  if (CONTEXT_OVERFLOW_PATTERNS.some((pattern) => pattern.test(errorBody))) {
    return true
  }
  if (TRANSIENT_OUTAGE_PATTERNS.some((pattern) => pattern.test(errorBody))) {
    return false
  }

  // Copilot's backend hangs rather than cleanly rejecting in the ~2.5-5.3 MB
  // dead zone, which Bun surfaces as a 500 or an upstream timeout. Restrict
  // that inference to plain 500s: 502/503/504 are gateway-level outages that
  // say nothing about whether the prompt fit.
  const isDeadZoneHang =
    (response.status === 500 || /operation timed out/i.test(errorBody))
    && bodyLength > 2_000_000

  return isDeadZoneHang
}

interface UpstreamTokenCounts {
  limit?: number
  used?: number
}

/**
 * Pull the token counts Copilot actually reported out of an error body, so the
 * client sees upstream's numbers rather than a local guess. Covers the phrasings
 * Copilot has emitted: "maximum context length is N tokens ... resulted in M
 * tokens", "exceeds the limit of N", and JSON bodies carrying explicit fields.
 */
function parseUpstreamTokenCounts(errorBody: string): UpstreamTokenCounts {
  const counts: UpstreamTokenCounts = {}

  const maxContext = /maximum context length is (\d+) tokens/i.exec(errorBody)
  if (maxContext) counts.limit = Number(maxContext[1])

  const resulted = /(?:resulted in|you requested|requested) (\d+) tokens/i.exec(
    errorBody,
  )
  if (resulted) counts.used = Number(resulted[1])

  const exceedsLimit = /exceeds the limit of (\d+)/i.exec(errorBody)
  if (exceedsLimit) counts.limit ??= Number(exceedsLimit[1])

  const promptTokens = /(\d+) prompt tokens/i.exec(errorBody)
  if (promptTokens) counts.used ??= Number(promptTokens[1])

  return counts
}

function sanitizeResponsesPayload(
  payload: ResponsesApiRequest,
): ResponsesApiRequest {
  let sanitized = normalizeToolDescriptions(
    sanitizeInvalidOutputImages(payload),
  )

  // Codex sends ChatGPT-only fast mode metadata; Copilot Responses rejects it.
  delete (sanitized as ResponsesApiRequest & { service_tier?: unknown })
    .service_tier

  if (requiresGpt6ResponsesSanitization(sanitized.model)) {
    sanitized = sanitizeGpt6Payload(sanitized)
  }

  if (!sanitized.tools?.some((tool) => tool.type === "image_generation")) {
    return sanitized
  }

  return {
    ...sanitized,
    tools: sanitized.tools.filter((tool) => tool.type !== "image_generation"),
  }
}

function requiresGpt6ResponsesSanitization(modelId: string): boolean {
  return modelId === "gpt-6-astra" || modelId === "gpt-6.1-sol"
}

function sanitizeGpt6Payload(
  payload: ResponsesApiRequest,
): ResponsesApiRequest {
  const sanitized = { ...payload }
  delete sanitized.temperature
  delete sanitized.top_p
  delete (sanitized as ResponsesApiRequest & { top_logprobs?: unknown })
    .top_logprobs
  delete (sanitized as ResponsesApiRequest & { logprobs?: unknown }).logprobs

  if (sanitized.include?.includes("message.output_text.logprobs")) {
    sanitized.include = sanitized.include.filter(
      (item) => item !== "message.output_text.logprobs",
    )
  }

  return sanitized
}

function normalizeToolDescriptions(
  payload: ResponsesApiRequest,
): ResponsesApiRequest {
  const tools = normalizeToolList(payload.tools)
  let sanitized = tools === payload.tools ? payload : { ...payload, tools }

  if (typeof payload.input === "string") return sanitized

  let input: Array<ResponsesInputItem> | undefined
  for (const [index, item] of payload.input.entries()) {
    const itemTools = normalizeToolList(item.tools, true)
    if (itemTools === item.tools) continue

    input ??= [...payload.input]
    input[index] = { ...item, tools: itemTools }
  }

  if (input) sanitized = { ...sanitized, input }
  return sanitized
}

function normalizeToolList(
  tools: ResponsesApiRequest["tools"],
  descriptionRequired = false,
): ResponsesApiRequest["tools"] {
  if (
    !Array.isArray(tools)
    || !tools.some(
      (tool) =>
        (descriptionRequired && !isNonBlankString(tool.description))
        || (tool.description !== undefined
          && !isNonBlankString(tool.description))
        || (tool.function?.description !== undefined
          && !isNonBlankString(tool.function.description)),
    )
  ) {
    return tools
  }

  return tools.map((tool) => {
    const sanitizedTool = { ...tool }
    if (descriptionRequired && !isNonBlankString(sanitizedTool.description)) {
      const nestedDescription = sanitizedTool.function?.description
      let name: string | undefined
      if (isNonBlankString(sanitizedTool.name)) {
        name = sanitizedTool.name
      } else if (isNonBlankString(sanitizedTool.function?.name)) {
        name = sanitizedTool.function.name
      }
      sanitizedTool.description = name ? `Tool ${name}` : "Tool"
      if (isNonBlankString(nestedDescription)) {
        sanitizedTool.description = nestedDescription
      }
    } else if (
      sanitizedTool.description !== undefined
      && !isNonBlankString(sanitizedTool.description)
    ) {
      delete sanitizedTool.description
    }
    if (
      sanitizedTool.function?.description !== undefined
      && !isNonBlankString(sanitizedTool.function.description)
    ) {
      sanitizedTool.function = { ...sanitizedTool.function }
      delete sanitizedTool.function.description
    }
    return sanitizedTool
  })
}

function isNonBlankString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0
}

function sanitizeInvalidOutputImages(
  payload: ResponsesApiRequest,
): ResponsesApiRequest {
  if (typeof payload.input === "string") return payload

  let input: Array<ResponsesInputItem> | undefined
  for (const [index, item] of payload.input.entries()) {
    if (!Array.isArray(item.output)) continue

    const sanitizedOutput = sanitizeOutputParts(item.output)
    if (sanitizedOutput === item.output) continue

    input ??= [...payload.input]
    input[index] = { ...item, output: sanitizedOutput }
  }

  return input ? { ...payload, input } : payload
}

function sanitizeOutputParts(
  output: NonNullable<ResponsesInputItem["output"]>,
): ResponsesInputItem["output"] {
  if (typeof output === "string") return output

  const sanitizedOutput = output.map((part) => sanitizeOutputPart(part))
  const changed = sanitizedOutput.some((part, index) => part !== output[index])

  return changed ? sanitizedOutput : output
}

function sanitizeOutputPart(part: ResponsesContentPart): ResponsesContentPart {
  if (hasInvalidImageUrl(part)) {
    return {
      type: "input_text",
      text: INVALID_OUTPUT_IMAGE_PLACEHOLDER,
    }
  }

  if (typeof part.image_url === "string" && part.type !== "input_image") {
    return {
      type: "input_image",
      image_url: part.image_url,
      detail: part.detail,
    }
  }

  if (part.type === "output_text") {
    return {
      type: "input_text",
      text: part.text ?? "",
    }
  }

  return part
}

function hasInvalidImageUrl(part: { image_url?: unknown }): boolean {
  if (
    !("image_url" in part)
    || part.image_url === null
    || part.image_url === undefined
  ) {
    return false
  }
  return typeof part.image_url !== "string" || !isValidImageUrl(part.image_url)
}

function isValidImageUrl(value: string): boolean {
  // MCP tools return inline images as base64 data URLs.
  if (/^data:image\/[a-z\d.+-]+;base64,./i.test(value)) return true

  try {
    const url = new URL(value)
    return url.protocol === "http:" || url.protocol === "https:"
  } catch {
    return false
  }
}

function responsesPayloadHasImages(payload: ResponsesApiRequest): boolean {
  if (typeof payload.input === "string") return false

  return payload.input.some((item) =>
    [item.content, item.output].some(
      (parts) =>
        Array.isArray(parts)
        && parts.some((part) => part.type === "input_image"),
    ),
  )
}
