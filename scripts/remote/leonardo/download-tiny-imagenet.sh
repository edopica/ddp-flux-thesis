#!/usr/bin/env bash
#
# download-tiny-imagenet.sh
#
# Idempotent download + extraction + verification of the Tiny ImageNet-200
# dataset (Stanford CS231n) for the ddp-flux-thesis benchmark.
#
# Leonardo-specific constraint: login nodes have outbound HTTPS, compute nodes
# have NO internet. Run this script on a login node only.
#
# Behaviour:
#   * The archive is downloaded with `curl -fL --retry 3 --retry-delay 5 -C -`
#     into a `.part` file and atomically renamed after size/hash checks.
#   * A re-run skips the download when the existing zip's SHA-256 matches the
#     recorded hash (env TINY_IMAGENET_SHA256, else first SHA-256 found in the
#     metadata file). A good zip is never re-downloaded.
#   * Extraction (`unzip -q -o` into the datasets dir) is skipped when the
#     sentinel <extract dir>/wnids.txt exists. No user data is ever deleted.
#   * The full verification always re-runs; any mismatch fails the script.
#   * The metadata file is regenerated (atomically) after every successful
#     run; the "Downloaded (UTC)" field is the archive file mtime.
#
# Usage:
#   bash ~/leonardo-setup/download-tiny-imagenet.sh
#
# Overridable environment variables:
#   WORK                          base work directory (default: /leonardo_work/IscrC_UNMASKED)
#   TINY_IMAGENET_DATASETS_DIR    parent of the extracted tree (default: $WORK/epicazio/datasets)
#   TINY_IMAGENET_DOWNLOADS_DIR   directory holding the .zip (default: <datasets dir>/downloads)
#   TINY_IMAGENET_METADATA        metadata file (default: <datasets dir>/tiny-imagenet-200-metadata.md)
#   TINY_IMAGENET_URL             source URL
#   TINY_IMAGENET_SHA256          trusted SHA-256 (overrides the metadata file for skip logic)
#   TINY_IMAGENET_BYTES           expected archive size in bytes
#   TINY_IMAGENET_PUBLISHED_MD5   community-reported MD5, informational cross-check
#
set -euo pipefail

# ---------------------------------------------------------------- configuration
WORK="${WORK:-/leonardo_work/IscrC_UNMASKED}"
EPICAZIO="${WORK}/epicazio"

DATASETS_DIR="${TINY_IMAGENET_DATASETS_DIR:-${EPICAZIO}/datasets}"
DOWNLOADS_DIR="${TINY_IMAGENET_DOWNLOADS_DIR:-${DATASETS_DIR}/downloads}"
ZIP="${DOWNLOADS_DIR}/tiny-imagenet-200.zip"
EXTRACT_DIR="${DATASETS_DIR}/tiny-imagenet-200"
METADATA="${TINY_IMAGENET_METADATA:-${DATASETS_DIR}/tiny-imagenet-200-metadata.md}"

URL="${TINY_IMAGENET_URL:-https://cs231n.stanford.edu/tiny-imagenet-200.zip}"
EXPECTED_BYTES="${TINY_IMAGENET_BYTES:-248100043}"

# Community-reported MD5 for tiny-imagenet-200.zip, widely mirrored (e.g. SJTU,
# GitHub tooling). NOT published on the official CS231n page; informational only.
PUBLISHED_MD5="${TINY_IMAGENET_PUBLISHED_MD5:-90528d7ca1a48142e341f4ef8d21d0de}"

SENTINEL="${EXTRACT_DIR}/wnids.txt"
TRAIN_DIR="${EXTRACT_DIR}/train"
VAL_DIR="${EXTRACT_DIR}/val"
TEST_DIR="${EXTRACT_DIR}/test"

# Recorded trusted hash: env var wins, else first 64-hex string in the metadata.
RECORDED_SHA256="${TINY_IMAGENET_SHA256:-}"
if [[ -z "${RECORDED_SHA256}" && -f "${METADATA}" ]]; then
  RECORDED_SHA256="$(grep -Eo '[0-9a-f]{64}' "${METADATA}" | head -n 1 || true)"
fi

# values captured by verification, used by summary and metadata
V_WNIDS=""
V_WORDS=""
V_TRAIN_DIRS=""
V_TRAIN_TOTAL=""
V_TRAIN_MIN=""
V_TRAIN_MAX=""
V_TRAIN_BOXES=""
V_VAL_IMGS=""
V_VAL_LINES=""
V_VAL_PRESENT=""
V_VAL_CLASSES=""
V_VAL_MIN=""
V_VAL_MAX=""
V_TEST_IMGS=""
ZIP_SHA256=""
ZIP_BYTES=""
ZIP_DU=""
ZIP_DU_BYTES=""
EXTRACT_DU=""
EXTRACT_DU_BYTES=""
MD5_STATUS="not checked"
MD5_ACTUAL=""
UNZIP_TEST_SUMMARY=""

# ---------------------------------------------------------------- helpers
log()  { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
ok()   { printf '  OK   %-46s %s\n' "$1" "$2"; }
fail() { printf '[%s] FAIL: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; exit 1; }

expect() { # label expected actual
  local label="$1" expected="$2" actual="$3"
  if [[ "${expected}" != "${actual}" ]]; then
    fail "verify: ${label} expected=${expected} actual=${actual}"
  fi
  ok "${label}" "${actual}"
}

# ---------------------------------------------------------------- download
download_zip() {
  local part="${ZIP}.part" attempt
  mkdir -p "${DOWNLOADS_DIR}"
  for attempt in 1 2; do
    log "download attempt ${attempt}: curl -fL --retry 3 --retry-delay 5 -C - -o ${part} ${URL}"
    if curl -fL --retry 3 --retry-delay 5 -C - -o "${part}" "${URL}"; then
      break
    fi
    if [[ "${attempt}" -eq 1 && -s "${part}" ]]; then
      log "WARN: resumed download failed; removing partial file and retrying from scratch"
      rm -f "${part}"
      continue
    fi
    fail "download failed after ${attempt} attempt(s)"
  done

  local size sha
  size="$(stat -c %s "${part}")"
  sha="$(sha256sum "${part}" | awk '{print $1}')"
  log "downloaded: ${size} bytes, sha256=${sha}"
  if [[ "${size}" != "${EXPECTED_BYTES}" ]]; then
    fail "download size mismatch: got ${size} bytes, expected ${EXPECTED_BYTES}"
  fi
  if [[ -n "${RECORDED_SHA256}" && "${sha}" != "${RECORDED_SHA256}" ]]; then
    fail "download SHA-256 mismatch: got ${sha}, recorded ${RECORDED_SHA256}"
  fi
  mv -f "${part}" "${ZIP}"
  log "[DONE] download: ${ZIP} (atomic rename from .part)"
}

ensure_zip() {
  if [[ -f "${ZIP}" ]]; then
    local have
    have="$(sha256sum "${ZIP}" | awk '{print $1}')"
    if [[ -n "${RECORDED_SHA256}" && "${have}" == "${RECORDED_SHA256}" ]]; then
      log "[DONE] download: skipped (existing zip sha256 matches recorded ${RECORDED_SHA256})"
      return 0
    fi
    if [[ -z "${RECORDED_SHA256}" ]]; then
      log "WARN: zip exists but no recorded SHA-256 (env unset, ${METADATA} missing or without hash); downloading a fresh copy"
    else
      log "WARN: existing zip sha256=${have} does not match recorded ${RECORDED_SHA256}; downloading a fresh copy"
    fi
  else
    log "zip not present; starting download"
  fi
  download_zip
}

# ---------------------------------------------------------------- integrity
test_zip_integrity() {
  local out rc=0
  log "zip integrity: unzip -tq ${ZIP}"
  out="$(unzip -tq "${ZIP}" 2>&1)" || rc=$?
  if [[ -n "${out}" ]]; then printf '%s\n' "${out}"; fi
  if [[ "${rc}" -ne 0 ]]; then
    fail "zip integrity: unzip -t reported errors (rc=${rc})"
  fi
  UNZIP_TEST_SUMMARY="$(printf '%s' "${out}" | tail -n 1)"
  if [[ -z "${UNZIP_TEST_SUMMARY}" ]]; then UNZIP_TEST_SUMMARY="no errors (rc=0, quiet mode)"; fi
  log "[DONE] zip integrity: OK (${UNZIP_TEST_SUMMARY})"
}

# ---------------------------------------------------------------- extract
extract_zip() {
  if [[ -f "${SENTINEL}" ]]; then
    log "[DONE] extract: skipped (sentinel present: ${SENTINEL})"
    return 0
  fi
  log "extracting: unzip -q -o ${ZIP} -d ${DATASETS_DIR}"
  unzip -q -o "${ZIP}" -d "${DATASETS_DIR}"
  if [[ ! -f "${SENTINEL}" ]]; then
    fail "extract: sentinel ${SENTINEL} missing after extraction"
  fi
  log "[DONE] extract: ${EXTRACT_DIR}"
}

# ---------------------------------------------------------------- verification
verify_basics() {
  [[ -f "${EXTRACT_DIR}/wnids.txt" ]] || fail "missing ${EXTRACT_DIR}/wnids.txt"
  [[ -f "${EXTRACT_DIR}/words.txt" ]] || fail "missing ${EXTRACT_DIR}/words.txt"
  V_WNIDS="$(wc -l < "${EXTRACT_DIR}/wnids.txt")"
  expect "wnids.txt lines" 200 "${V_WNIDS}"
  local wnids_uniq
  wnids_uniq="$(sort -u "${EXTRACT_DIR}/wnids.txt" | wc -l)"
  expect "wnids.txt unique entries" 200 "${wnids_uniq}"
  V_WORDS="$(wc -l < "${EXTRACT_DIR}/words.txt")"
  if (( V_WORDS < 200 )); then
    fail "verify: words.txt lines expected >= 200, got ${V_WORDS}"
  fi
  ok "words.txt class lines (>= 200)" "${V_WORDS}"
}

verify_train() {
  [[ -d "${TRAIN_DIR}" ]] || fail "missing ${TRAIN_DIR}"
  V_TRAIN_DIRS="$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  expect "train class directories" 200 "${V_TRAIN_DIRS}"

  local d n anomalies="" total=0 tmin=1000000000 tmax=0
  while IFS= read -r d; do
    n="$(find "${d}/images" -maxdepth 1 -type f -iname '*.jpeg' | wc -l)"
    total=$(( total + n ))
    if [[ "${n}" -ne 500 ]]; then anomalies="${anomalies}$(basename "${d}")=${n} "; fi
    if (( n < tmin )); then tmin=${n}; fi
    if (( n > tmax )); then tmax=${n}; fi
  done < <(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ -n "${anomalies}" ]]; then
    fail "verify: train classes with != 500 JPEGs: ${anomalies}"
  fi
  V_TRAIN_TOTAL="${total}"
  V_TRAIN_MIN="${tmin}"
  V_TRAIN_MAX="${tmax}"
  expect "train JPEG images" 100000 "${V_TRAIN_TOTAL}"
  ok "train images per class (min/max)" "${V_TRAIN_MIN}/${V_TRAIN_MAX}"

  V_TRAIN_BOXES="$(find "${TRAIN_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*_boxes.txt' | wc -l)"
  expect "train *_boxes.txt files" 200 "${V_TRAIN_BOXES}"

  local missing
  missing="$(comm -23 <(sort "${EXTRACT_DIR}/wnids.txt") \
                       <(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
             | paste -sd, - || true)"
  if [[ -n "${missing}" ]]; then
    fail "verify: wnids.txt entries missing under train/: ${missing}"
  fi
  ok "all wnids.txt entries present under train/" "yes"
}

verify_val() {
  [[ -d "${VAL_DIR}/images" ]] || fail "missing ${VAL_DIR}/images"
  [[ -f "${VAL_DIR}/val_annotations.txt" ]] || fail "missing ${VAL_DIR}/val_annotations.txt"
  V_VAL_IMGS="$(find "${VAL_DIR}/images" -maxdepth 1 -type f -iname '*.jpeg' | wc -l)"
  expect "val JPEG images" 10000 "${V_VAL_IMGS}"
  V_VAL_LINES="$(wc -l < "${VAL_DIR}/val_annotations.txt")"
  expect "val_annotations.txt lines" 10000 "${V_VAL_LINES}"

  local present=0 fname rest
  while IFS=$'\t' read -r fname rest; do
    [[ -n "${fname}" ]] || continue
    if [[ -f "${VAL_DIR}/images/${fname}" ]]; then present=$(( present + 1 )); fi
  done < "${VAL_DIR}/val_annotations.txt"
  V_VAL_PRESENT="${present}"
  expect "annotated filenames present under val/images" 10000 "${V_VAL_PRESENT}"

  local minclass maxclass nclasses
  read -r minclass maxclass nclasses < <(awk -F'\t' '{c[$2]++} END { min=-1; max=0; n=0; for (k in c) { n++; if (min<0 || c[k]<min) min=c[k]; if (c[k]>max) max=c[k] } print min, max, n }' "${VAL_DIR}/val_annotations.txt")
  V_VAL_CLASSES="${nclasses}"
  V_VAL_MIN="${minclass}"
  V_VAL_MAX="${maxclass}"
  ok "val classes in annotations" "${V_VAL_CLASSES}"
  expect "val images per class (min)" 50 "${V_VAL_MIN}"
  expect "val images per class (max)" 50 "${V_VAL_MAX}"
  ok "val images per class (min/max)" "${V_VAL_MIN}/${V_VAL_MAX}"
}

verify_test() {
  [[ -d "${TEST_DIR}/images" ]] || fail "missing ${TEST_DIR}/images"
  V_TEST_IMGS="$(find "${TEST_DIR}/images" -maxdepth 1 -type f -iname '*.jpeg' | wc -l)"
  expect "test JPEG images" 10000 "${V_TEST_IMGS}"
}

verify_dataset() {
  log "== verification =="
  verify_basics
  verify_train
  verify_val
  verify_test
  log "[DONE] verify: all checks passed"
}

# ---------------------------------------------------------------- summary
print_summary() {
  local sha_line
  sha_line="$(sha256sum "${ZIP}")"
  printf '%s\n' "${sha_line}"
  ZIP_SHA256="${sha_line%% *}"
  ZIP_BYTES="$(stat -c %s "${ZIP}")"
  log "zip byte size (stat -c %s): ${ZIP_BYTES}"
  ZIP_DU="$(du -sh "${ZIP}" | awk '{print $1}')"
  ZIP_DU_BYTES="$(du -sb "${ZIP}" | awk '{print $1}')"
  EXTRACT_DU="$(du -sh "${EXTRACT_DIR}" | awk '{print $1}')"
  EXTRACT_DU_BYTES="$(du -sb "${EXTRACT_DIR}" | awk '{print $1}')"
  log "du -sh zip: ${ZIP_DU} (${ZIP_DU_BYTES} bytes apparent)"
  log "du -sh extracted: ${EXTRACT_DU} (${EXTRACT_DU_BYTES} bytes apparent)"

  if [[ -n "${PUBLISHED_MD5}" ]]; then
    MD5_ACTUAL="$(md5sum "${ZIP}" | awk '{print $1}')"
    if [[ "${MD5_ACTUAL}" == "${PUBLISHED_MD5}" ]]; then
      MD5_STATUS="MATCH"
    else
      MD5_STATUS="MISMATCH"
    fi
    log "published MD5 cross-check: ${MD5_STATUS} (actual=${MD5_ACTUAL} published=${PUBLISHED_MD5})"
  else
    MD5_STATUS="not checked (no published MD5 provided)"
  fi

  cat <<EOF

================== TINY IMAGENET-200 VERIFICATION SUMMARY ==================
zip path       : ${ZIP}
zip byte size  : ${ZIP_BYTES} (du -sh ${ZIP_DU}, du -sb ${ZIP_DU_BYTES} B)
zip SHA-256    : ${ZIP_SHA256}
published MD5  : ${MD5_STATUS}
extract path   : ${EXTRACT_DIR}
extract size   : du -sh ${EXTRACT_DU} (du -sb ${EXTRACT_DU_BYTES} B)
unzip -t       : OK
train          : ${V_TRAIN_TOTAL} JPEG in ${V_TRAIN_DIRS} classes (min/max per class ${V_TRAIN_MIN}/${V_TRAIN_MAX}), ${V_TRAIN_BOXES} *_boxes.txt
val            : ${V_VAL_IMGS} JPEG, ${V_VAL_LINES} annotation lines, ${V_VAL_CLASSES} classes (min/max per class ${V_VAL_MIN}/${V_VAL_MAX})
test           : ${V_TEST_IMGS} JPEG
wnids.txt      : ${V_WNIDS} lines
words.txt      : ${V_WORDS} lines
============================================================================
TINY_IMAGENET_OK train=100000 val=10000 test=10000 classes=200
EOF
}

# ---------------------------------------------------------------- metadata
write_metadata() {
  local ts verify_ts
  ts="$(date -u -d "@$(stat -c %Y "${ZIP}")" '+%Y-%m-%dT%H:%M:%SZ')"
  verify_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  {
    cat <<EOF
# Tiny ImageNet-200 dataset metadata

- Dataset: Tiny ImageNet-200 (Stanford CS231n)
- URL: ${URL}
- Downloaded (UTC): ${ts}
- Archive: ${ZIP}
- Archive byte size: ${ZIP_BYTES}
- SHA-256: ${ZIP_SHA256}
- SHA-256 source: self-computed on the Leonardo login node (the official CS231n page publishes no checksum)
- Published MD5 cross-check: ${MD5_STATUS} (community-reported MD5 \`${PUBLISHED_MD5}\`, actual \`${MD5_ACTUAL}\`)
- Extracted to: ${EXTRACT_DIR}
- Disk usage: archive \`du -sh\` ${ZIP_DU} (${ZIP_DU_BYTES} B apparent); extracted \`du -sh\` ${EXTRACT_DU} (${EXTRACT_DU_BYTES} B apparent)
- Last verified (UTC): ${verify_ts}

## Verified counts (all checks passed, \`unzip -t\` OK)

| Item | Expected | Observed |
|---|---|---|
| wnids.txt lines | 200 | ${V_WNIDS} |
| words.txt lines | >= 200 | ${V_WORDS} |
| train class directories | 200 | ${V_TRAIN_DIRS} |
| train JPEG images | 100000 | ${V_TRAIN_TOTAL} |
| train JPEG per class (min/max) | 500/500 | ${V_TRAIN_MIN}/${V_TRAIN_MAX} |
| train *_boxes.txt files | 200 | ${V_TRAIN_BOXES} |
| val JPEG images | 10000 | ${V_VAL_IMGS} |
| val_annotations.txt lines | 10000 | ${V_VAL_LINES} |
| annotated files present under val/images | 10000 | ${V_VAL_PRESENT} |
| val classes in annotations | 200 | ${V_VAL_CLASSES} |
| val JPEG per class (min/max) | 50/50 | ${V_VAL_MIN}/${V_VAL_MAX} |
| test JPEG images | 10000 | ${V_TEST_IMGS} |

## Commands used

Staging from the thesis repo root (local machine):

    ssh leonardo 'mkdir -p ~/leonardo-setup'
    ssh leonardo 'cat > ~/leonardo-setup/download-tiny-imagenet.sh' < scripts/remote/leonardo/download-tiny-imagenet.sh

Execution (Leonardo login node):

    ssh leonardo 'bash ~/leonardo-setup/download-tiny-imagenet.sh'

Key commands inside the script (first run):

    curl -fL --retry 3 --retry-delay 5 -C - -o ${ZIP}.part ${URL}
    mv -f ${ZIP}.part ${ZIP}
    unzip -tq ${ZIP}
    unzip -q -o ${ZIP} -d ${DATASETS_DIR}
    sha256sum ${ZIP}
    stat -c %s ${ZIP}
    du -sh ${EXTRACT_DIR}
    du -sh ${ZIP}

Idempotency: a re-run skips the download when \`sha256sum ${ZIP}\` equals the
recorded SHA-256 above and skips extraction when \`${SENTINEL}\` exists; the
verification always re-runs.
EOF
  } > "${METADATA}.tmp"
  mv -f "${METADATA}.tmp" "${METADATA}"
  log "[DONE] metadata: ${METADATA}"
}

# ---------------------------------------------------------------- main
main() {
  log "== tiny-imagenet-200 setup =="
  log "url           : ${URL}"
  log "zip           : ${ZIP}"
  log "extract dir   : ${EXTRACT_DIR}"
  log "metadata      : ${METADATA}"
  log "recorded sha  : ${RECORDED_SHA256:-<none>}"
  log "expected size : ${EXPECTED_BYTES} bytes"

  command -v curl >/dev/null     || fail "curl not found"
  command -v unzip >/dev/null    || fail "unzip not found"
  command -v sha256sum >/dev/null || fail "sha256sum not found"

  mkdir -p "${DOWNLOADS_DIR}" "${DATASETS_DIR}"

  ensure_zip
  test_zip_integrity
  extract_zip
  verify_dataset
  print_summary
  write_metadata
  log "ALL DONE"
}

main "$@"
