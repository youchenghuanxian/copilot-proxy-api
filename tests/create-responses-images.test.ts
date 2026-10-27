import { afterEach, expect, mock, test } from "bun:test"

import type {
  ResponsesApiRequest,
  ResponsesInputItem,
} from "~/routes/responses/types"

import { state } from "~/lib/state"
import { createResponses } from "~/services/copilot/create-responses"

const originalFetch = globalThis.fetch
state.copilotToken = "test-token"
state.vsCodeVersion = "1.0.0"
state.accountType = "individual"

afterEach(() => {
  mock.restore()
  globalThis.fetch = originalFetch
})

function bodyToString(body: unknown): string {
  if (typeof body !== "string") {
    throw new TypeError("expected fetch body to be a string")
  }
  return body
}

async function forwardResponsesInput(input: Array<ResponsesInputItem>) {
  const fetchMock = mock(
    (_url: string, opts: RequestInit) => new Response(bodyToString(opts.body)),
  )
  globalThis.fetch = fetchMock as unknown as typeof fetch
  await createResponses({ model: "gpt-5.5", input })
  const request = fetchMock.mock.calls[0][1]
  const forwarded = JSON.parse(
    bodyToString(request.body),
  ) as ResponsesApiRequest
  return {
    input: Array.isArray(forwarded.input) ? forwarded.input : [],
    headers: new Headers(request.headers),
  }
}

const dataImageUrl =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/C9sAAAAASUVORK5CYII="

test.each([
  ["function_call", "input_image", dataImageUrl],
  ["custom_tool_call", "input_image", dataImageUrl],
  ["function_call", "output_text", "http://example.com/image.png"],
  ["custom_tool_call", "output_text", "https://example.com/image.png"],
])(
  "preserves Responses %s output images of type %s and enables vision",
  async (callType, partType, imageUrl) => {
    const { input, headers } = await forwardResponsesInput([
      {
        type: callType,
        call_id: "call_image",
        name: "capture_screenshot",
        ...(callType === "function_call" ?
          { arguments: "{}" }
        : { input: "capture" }),
      },
      {
        type: `${callType}_output`,
        call_id: "call_image",
        content: [{ type: "input_text", text: "Screenshot result" }],
        output: [{ type: partType, image_url: imageUrl, detail: "high" }],
      },
    ])
    expect(input[1].output).toEqual([
      { type: "input_image", image_url: imageUrl, detail: "high" },
    ])
    expect(headers.get("copilot-vision-request")).toBe("true")
  },
)

test("enables vision for Responses message content images", async () => {
  const content = [{ type: "input_image", image_url: dataImageUrl }]
  const { input, headers } = await forwardResponsesInput([
    { role: "user", content },
  ])
  expect(input[0].content).toEqual(content)
  expect(headers.get("copilot-vision-request")).toBe("true")
})

test("does not enable vision for string or array Responses text tool outputs", async () => {
  const original: Array<ResponsesInputItem> = [
    {
      type: "function_call_output",
      call_id: "call_text",
      output: "Screenshot saved",
    },
    {
      type: "custom_tool_call_output",
      call_id: "call_parts",
      output: [
        {
          type: "input_text",
          text: "Image URL: https://example.com/image.png",
        },
      ],
    },
  ]
  const { input, headers } = await forwardResponsesInput(original)
  expect(input).toEqual(original)
  expect(headers.has("copilot-vision-request")).toBe(false)
})

test("replaces invalid and non-image data URLs in Responses tool outputs without enabling vision", async () => {
  const imageUrls = [
    "data:text/plain;base64,aGVsbG8=",
    "data:application/pdf;base64,JVBERi0=",
    "data:image/png,not-base64",
    "data:image/;base64,aGVsbG8=",
    "data:image/png;base64,",
  ]
  const { input, headers } = await forwardResponsesInput([
    {
      type: "function_call_output",
      call_id: "call_invalid_images",
      output: imageUrls.map((imageUrl) => ({
        type: "input_image",
        image_url: imageUrl,
      })),
    },
  ])
  expect(input[0].output).toEqual(
    imageUrls.map(() => ({
      type: "input_text",
      text: "[image output removed because its URL is not valid for Copilot Responses]",
    })),
  )
  expect(headers.has("copilot-vision-request")).toBe(false)
})
