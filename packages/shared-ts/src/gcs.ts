/** Shared GCS helpers used by API + compose + record-mix stages.
 *
 * Dev mode: if `DEV_FS_ROOT` is set, read/write go to the local filesystem
 * under that root instead of GCS. Keeps the TS + Python halves consistent
 * (see packages/shared-py/shared/gcs.py for the mirror).
 */

import { promises as fsp } from "node:fs";
import { dirname, join, resolve as pathResolve } from "node:path";
import { Storage } from "@google-cloud/storage";
import { required } from "./env.js";

let _storage: Storage | undefined;

function devRoot(): string | undefined {
  const v = process.env.DEV_FS_ROOT;
  return v && v.length > 0 ? v : undefined;
}

export function storage(): Storage {
  if (!_storage) _storage = new Storage();
  return _storage;
}

export function bucket() {
  return storage().bucket(required("GCS_BUCKET"));
}

/** V4 signed PUT URL for direct browser upload. 15-min default expiry. */
export async function signedPutUrl(
  objectPath: string,
  contentType: string,
  expiresInSeconds = 900,
): Promise<string> {
  const [url] = await bucket()
    .file(objectPath)
    .getSignedUrl({
      version: "v4",
      action: "write",
      expires: Date.now() + expiresInSeconds * 1000,
      contentType,
    });
  return url;
}

export function publicUrl(objectPath: string): string {
  const root = devRoot();
  if (root) return `file://${pathResolve(root, objectPath)}`;
  const b = required("GCS_BUCKET");
  return `https://storage.googleapis.com/${b}/${objectPath}`;
}

export async function objectExists(objectPath: string): Promise<boolean> {
  const root = devRoot();
  if (root) {
    try {
      await fsp.access(join(root, objectPath));
      return true;
    } catch {
      return false;
    }
  }
  const [exists] = await bucket().file(objectPath).exists();
  return exists;
}

export async function uploadFile(
  objectPath: string,
  localPath: string,
  contentType: string,
): Promise<string> {
  const root = devRoot();
  if (root) {
    const dst = join(root, objectPath);
    await fsp.mkdir(dirname(dst), { recursive: true });
    await fsp.copyFile(localPath, dst);
    return publicUrl(objectPath);
  }
  await bucket().upload(localPath, {
    destination: objectPath,
    metadata: { contentType },
  });
  return publicUrl(objectPath);
}

export async function uploadBuffer(
  objectPath: string,
  body: Buffer | string,
  contentType: string,
): Promise<string> {
  const root = devRoot();
  if (root) {
    const dst = join(root, objectPath);
    await fsp.mkdir(dirname(dst), { recursive: true });
    await fsp.writeFile(dst, body);
    return publicUrl(objectPath);
  }
  await bucket().file(objectPath).save(body, { contentType });
  return publicUrl(objectPath);
}

export async function downloadFile(objectPath: string, localPath: string): Promise<void> {
  const root = devRoot();
  if (root) {
    await fsp.mkdir(dirname(localPath), { recursive: true });
    await fsp.copyFile(join(root, objectPath), localPath);
    return;
  }
  await bucket().file(objectPath).download({ destination: localPath });
}

/** gs://bucket/path → path (strips bucket). Also accepts file:// in dev. */
export function objectPathFromGsUri(uri: string): string {
  if (uri.startsWith("file://")) return uri.slice("file://".length);
  const m = /^gs:\/\/[^/]+\/(.+)$/.exec(uri);
  if (!m) throw new Error(`expected gs:// URI, got ${uri}`);
  return m[1]!;
}
