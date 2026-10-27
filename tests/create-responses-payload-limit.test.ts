import { afterEach, beforeEach, expect, mock, test } from "bun:test"

import type { ResponsesApiRequest } from "~/routes/responses/types"

import { state } from "~/lib/state"
import { createResponses } from "~/services/copilot/create-responses"

let originalFetch = globalThis.fetch
let originalState = { ...state }

beforeEach(() => {
  originalFetch = globalThis.fetch
  originalState = { ...state }
  state.copilotToken = "test-token"
  state.vsCodeVersion = "1.0.0"
  state.accountType = "individual"
  state.responsesContextTrim = false
})

afterEach(() => {
  mock.restore()
  globalThis.fetch = originalFetch
  state.copilotToken = originalState.copilotToken
  state.vsCodeVersion = originalState.vsCodeVersion
  state.accountType = originalState.accountType
  state.responsesContextTrim = originalState.responsesContextTrim
  state.responsesMaxPayloadBytes = originalState.responsesMaxPayloadBytes
})

async function forwardPayload(payload: ResponsesApiRequest) {
  const fetchMock = mock(
    (_url: string, _opts: RequestInit) =>
      new Response(JSON.stringify({ id: "resp_payload_limit" }), {
        headers: { "content-type": "application/json" },
      }),
  )
  globalThis.fetch = fetchMock as unknown as typeof fetch

  await createResponses(payload)

  expect(fetchMock).toHaveBeenCalledTimes(1)
  const request = fetchMock.mock.calls[0][1]
  const body = request.body
  if (typeof body !== "string") {
    throw new TypeError("expected fetch body to be a string")
  }
  return {
    body,
    forwarded: JSON.parse(body) as ResponsesApiRequest,
    headers: new Headers(request.headers),
  }
}

function textPayload(
  inputKind: "string" | "array",
  text: string,
): ResponsesApiRequest {
  return {
    model: "unknown-payload-limit-model",
    input:
      inputKind === "string" ? text : (
        [{ role: "user", content: [{ type: "input_text", text }] }]
      ),
  }
}

function getInputText(payload: ResponsesApiRequest): string {
  if (typeof payload.input === "string") return payload.input
  const content = payload.input[0]?.content
  const text = Array.isArray(content) ? content[0]?.text : content
  if (typeof text !== "string") {
    throw new TypeError("expected forwarded text input")
  }
  return text
}

test("forwards more than 6 MB of images and history unchanged with the default 10 MB limit", async () => {
  const imageUrl = `data:image/png;base64,${"a".repeat(4_000_000)}`
  const payload: ResponsesApiRequest = {
    model: "unknown-payload-limit-model",
    input: [
      { role: "user", content: `Earlier task: ${"x".repeat(2_000_000)}` },
      { role: "assistant", content: "Inspecting the image." },
      {
        type: "function_call",
        call_id: "call_large_image",
        name: "capture_screenshot",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_large_image",
        output: [{ type: "input_image", image_url: imageUrl }],
      },
      {
        role: "user",
        content: "Explain the screenshot using the earlier task.",
      },
    ],
  }
  const originalInput = JSON.stringify(payload.input)

  const { body, forwarded, headers } = await forwardPayload(payload)

  expect(Buffer.byteLength(body, "utf8")).toBeGreaterThan(6_000_000)
  expect(Buffer.byteLength(body, "utf8")).toBeLessThanOrEqual(10_000_000)
  // Compare as a boolean to keep a regression from printing megabytes of data.
  expect(JSON.stringify(forwarded.input) === originalInput).toBe(true)
  expect(JSON.stringify(payload.input) === originalInput).toBe(true)
  expect(headers.get("copilot-vision-request")).toBe("true")
})

test.each(["string", "array"] as const)(
  "fits Chinese %s input using serialized UTF-8 bytes",
  async (inputKind) => {
    state.responsesMaxPayloadBytes = 1_000
    const latestTask = "请回答最后的问题"
    const payload = textPayload(inputKind, "中文".repeat(300) + latestTask)
    const originalBody = JSON.stringify(payload)

    expect(originalBody.length).toBeLessThan(1_000)
    expect(Buffer.byteLength(originalBody, "utf8")).toBeGreaterThan(1_000)

    const { body, forwarded } = await forwardPayload(payload)

    expect(Buffer.byteLength(body, "utf8")).toBeLessThanOrEqual(1_000)
    expect(getInputText(forwarded)).toContain("older response input truncated")
    expect(getInputText(forwarded).endsWith(latestTask)).toBe(true)
    expect(JSON.stringify(payload)).toBe(originalBody)
  },
)

test.each(["string", "array"] as const)(
  "includes JSON escaping when fitting %s input",
  async (inputKind) => {
    state.responsesMaxPayloadBytes = 1_000
    const latestTask = "Answer the latest task"
    const text = '\n"\\'.repeat(240) + latestTask
    const payload = textPayload(inputKind, text)
    const originalBody = JSON.stringify(payload)

    expect(text.length).toBeLessThan(1_000)
    expect(Buffer.byteLength(originalBody, "utf8")).toBeGreaterThan(1_000)

    const { body, forwarded } = await forwardPayload(payload)

    expect(Buffer.byteLength(body, "utf8")).toBeLessThanOrEqual(1_000)
    expect(getInputText(forwarded)).toContain("older response input truncated")
    expect(getInputText(forwarded).endsWith(latestTask)).toBe(true)
    expect(JSON.stringify(payload)).toBe(originalBody)
  },
)

test("preserves Unicode and escaped input that fits the configured byte limit", async () => {
  state.responsesMaxPayloadBytes = 2_500
  const payload = textPayload("array", '中文\n"\\'.repeat(180))
  const originalBody = JSON.stringify(payload)
  expect(Buffer.byteLength(originalBody, "utf8")).toBeLessThanOrEqual(2_500)

  const { forwarded } = await forwardPayload(payload)

  expect(forwarded.input).toEqual(payload.input)
})

test.each(["string", "array"] as const)(
  "keeps emoji intact when fitting %s input to a byte limit",
  async (inputKind) => {
    state.responsesMaxPayloadBytes = 1_000
    const payload = textPayload(inputKind, "😀".repeat(400))

    const { body, forwarded } = await forwardPayload(payload)
    const retainedText = getInputText(forwarded)

    expect(Buffer.byteLength(body, "utf8")).toBeLessThanOrEqual(1_000)
    expect(retainedText).toContain("older response input truncated")
    expect(retainedText.endsWith("😀")).toBe(true)
    expect(retainedText.isWellFormed()).toBe(true)
  },
)

test("keeps the configured byte limit when token-derived trimming is enabled", async () => {
  state.responsesContextTrim = true
  state.responsesMaxPayloadBytes = 1_000
  const payload = textPayload("array", `${"x".repeat(1_500)}latest task`)
  // This model has a known context window well above the configured byte cap.
  payload.model = "gpt-6.1-sol"

  const { body, forwarded } = await forwardPayload(payload)

  expect(Buffer.byteLength(body, "utf8")).toBeLessThanOrEqual(1_000)
  expect(getInputText(forwarded).endsWith("latest task")).toBe(true)
})
