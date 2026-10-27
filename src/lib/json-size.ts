import { Buffer } from "node:buffer"

export function jsonByteLength(value: object | string): number {
  return Buffer.byteLength(JSON.stringify(value), "utf8")
}

/** Keep the longest suffix that fits, including JSON escaping and the prefix. */
export function truncateJsonString(
  value: string,
  maxBytes: number,
  prefix: string,
): string {
  let result = prefix
  let low = 0
  let high = value.length

  while (low <= high) {
    const length = Math.floor((low + high) / 2)
    let start = value.length - length
    // Do not split a UTF-16 surrogate pair at the retained suffix boundary.
    if ((value.codePointAt(start - 1) ?? 0) > 0xffff) {
      start++
    }
    const candidate = prefix + value.slice(start)
    if (jsonByteLength(candidate) <= maxBytes) {
      result = candidate
      low = length + 1
    } else {
      high = length - 1
    }
  }

  return result
}
