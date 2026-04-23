/** Shared GCS helpers used by API + compose + record-mix stages. */

import { Storage } from "@google-cloud/storage";
import { required } from "./env.js";

let _storage: Storage | undefined;

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
  const b = required("GCS_BUCKET");
  return `https://storage.googleapis.com/${b}/${objectPath}`;
}

export async function objectExists(objectPath: string): Promise<boolean> {
  const [exists] = await bucket().file(objectPath).exists();
  return exists;
}

export async function uploadFile(
  objectPath: string,
  localPath: string,
  contentType: string,
): Promise<string> {
  await bucket().upload(localPath, {
    destination: objectPath,
    metadata: { contentType },
  });
  return publicUrl(objectPath);
}

export async function downloadFile(objectPath: string, localPath: string): Promise<void> {
  await bucket().file(objectPath).download({ destination: localPath });
}
