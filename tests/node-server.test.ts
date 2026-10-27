import { expect, test } from "bun:test"
import { fileURLToPath } from "node:url"

test("Node HTTP adapter preserves requests and propagates client disconnects", async () => {
  // Bun selects a different srvx adapter, so exercise the npm runtime in Node.
  const fixture = fileURLToPath(
    new URL("./fixtures/node-request-lifecycle.ts", import.meta.url),
  )
  const source = await Bun.file(fixture).text()
  const script = new Bun.Transpiler({
    loader: "ts",
    target: "node",
  }).transformSync(source)
  const child = Bun.spawn(["node", "--input-type=module", "--eval", script], {
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    stdout: "pipe",
    stderr: "pipe",
  })
  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ])
    expect({ exitCode, output: exitCode === 0 ? "" : stdout + stderr }).toEqual(
      {
        exitCode: 0,
        output: "",
      },
    )
  } finally {
    child.kill()
  }
}, 20_000)
