import { Redis } from "@upstash/redis";
import { required } from "./env.js";

let _redis: Redis | undefined;

/** Lazy singleton Upstash Redis client, reads from env on first call. */
export function redis(): Redis {
  if (!_redis) {
    _redis = new Redis({
      url: required("UPSTASH_REDIS_REST_URL"),
      token: required("UPSTASH_REDIS_REST_TOKEN"),
    });
  }
  return _redis;
}
