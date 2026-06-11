# Compose is pure TS (no ML). node:20-alpine stays lean even as a single-stage
# image — the whole pnpm workspace footprint for contracts+shared-ts+compose
# lands under ~200 MB, which beats wrangling pnpm deploy + workspace symlinks.
FROM node:20-alpine

RUN apk add --no-cache git \
 && corepack enable pnpm

WORKDIR /repo

# Manifests first so `pnpm install` is cacheable across code-only rebuilds.
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/contracts/package.json ./packages/contracts/
COPY packages/shared-ts/package.json ./packages/shared-ts/
COPY stages/compose/package.json ./stages/compose/

RUN pnpm install --frozen-lockfile --filter @annemusic/stage-compose...

# Sources second so edits don't bust the install layer.
COPY packages/contracts ./packages/contracts
COPY packages/shared-ts ./packages/shared-ts
COPY stages/compose ./stages/compose

RUN pnpm --filter @annemusic/contracts run build \
 && pnpm --filter @annemusic/shared-ts run build \
 && pnpm --filter @annemusic/stage-compose run build

WORKDIR /repo/stages/compose
ENV NODE_ENV=production
EXPOSE 8104
CMD ["node", "dist/main.js"]
