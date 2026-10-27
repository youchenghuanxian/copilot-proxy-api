import type { TestContext } from "node:test"
import type { ServerHandler } from "srvx"

import { Hono } from "hono"
import assert from "node:assert/strict"
import { once } from "node:events"
import { test } from "node:test"
import { setTimeout } from "node:timers/promises"
import { serve } from "srvx"

const options = { hostname: "127.0.0.1", port: 0, silent: true }
const body = JSON.stringify({ model: "test", input: "hello" })
const encoder = new TextEncoder()

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

async function createProxy(t: TestContext, upstreamHandler: ServerHandler) {
  const upstream = serve({ ...options, fetch: upstreamHandler })
  t.after(() => upstream.close(true))
  await upstream.ready()
  const upstreamUrl = upstream.url
  assert.ok(upstreamUrl)

  // Match the application boundary: Hono consumes JSON before forwarding the
  // incoming request's cancellation signal to the upstream fetch.
  const app = new Hono()
  app.post("/v1/responses", async (c) => {
    const payload = await c.req.json<unknown>()
    return fetch(upstreamUrl, {
      method: "POST",
      body: JSON.stringify(payload),
      signal: c.req.raw.signal,
    })
  })
  app.onError((error) => new Response(error.message, { status: 500 }))
  const proxy = serve({ ...options, fetch: app.fetch })
  t.after(() => proxy.close(true))
  await proxy.ready()
  return new URL("/v1/responses", proxy.url)
}

await test(
  "a complete POST body does not cancel a delayed JSON response",
  { timeout: 3000 },
  async (t) => {
    const url = await createProxy(t, async (request) => {
      const payload = (await request.json()) as { input: string }
      await setTimeout(30, undefined, { signal: request.signal })
      return Response.json({
        input: payload.input,
        aborted: request.signal.aborted,
      })
    })
    const response = await fetch(url, { method: "POST", body })
    assert.equal(response.status, 200)
    assert.deepEqual(await response.json(), { input: "hello", aborted: false })
  },
)

await test(
  "POST responses stream through completion after the upload ends",
  { timeout: 3000 },
  async (t) => {
    const events = [
      'data: {"type":"response.output_text.delta","delta":"hello"}\n\n',
      'data: {"type":"response.completed"}\n\n',
    ]
    const url = await createProxy(t, async (request) => {
      await request.json()
      let index = 0
      return new Response(
        new ReadableStream({
          async pull(controller) {
            await setTimeout(30, undefined, { signal: request.signal })
            controller.enqueue(encoder.encode(events[index++]))
            if (index === events.length) controller.close()
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    })
    const response = await fetch(url, { method: "POST", body })
    assert.equal(response.status, 200)
    assert.equal(await response.text(), events.join(""))
  },
)

await test(
  "disconnecting before response headers aborts the upstream fetch",
  { timeout: 3000 },
  async (t) => {
    const started = deferred<undefined>()
    const cancelled = deferred<boolean>()
    const url = await createProxy(t, async (request) => {
      await request.json()
      const signal = request.signal
      const aborted = once(signal, "abort")
      started.resolve(undefined)
      await aborted
      cancelled.resolve(signal.aborted)
      return new Response(null, { status: 204 })
    })
    const client = new AbortController()
    t.after(() => client.abort())
    const response = fetch(url, { method: "POST", body, signal: client.signal })
    const rejected = assert.rejects(response, { name: "AbortError" })
    await started.promise
    client.abort()
    await rejected
    assert.equal(await cancelled.promise, true)
  },
)

await test(
  "disconnecting during SSE cancels the upstream stream",
  { timeout: 3000 },
  async (t) => {
    const cancelled = deferred<boolean>()
    const url = await createProxy(t, async (request) => {
      await request.json()
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('data: {"delta":"hello"}\n\n'))
          },
          cancel() {
            cancelled.resolve(true)
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    })
    const client = new AbortController()
    t.after(() => client.abort())
    const response = await fetch(url, {
      method: "POST",
      body,
      signal: client.signal,
    })
    assert.equal(response.status, 200)
    assert.ok(response.body)
    const reader = response.body.getReader()
    assert.equal((await reader.read()).done, false)
    client.abort()
    await reader.cancel().catch(() => undefined)
    assert.equal(await cancelled.promise, true)
  },
)
