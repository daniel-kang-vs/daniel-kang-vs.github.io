#!/usr/bin/env bash
# Shared resume paths:
#   Resume Tailor/{company_slug}/{company_slug}_{YYYYMMDD}_{role_slug}_resume.{tex,pdf}
set -euo pipefail

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_|_$//g; s/_+/_/g'
}

resume_basename() {
  local company="${1:?company required}"
  local role="${2:?role required}"
  local date="${3:-$(date +%Y%m%d)}"

  local company_slug role_slug
  company_slug="$(slugify "$company")"
  role_slug="$(slugify "$role")"

  printf '%s_%s_%s_resume' "$company_slug" "$date" "$role_slug"
}

company_dir() {
  local root="${1:?root required}"
  local company="${2:?company required}"
  printf '%s/%s' "$root" "$(slugify "$company")"
}

resume_tex_path() {
  local root="${1:?root required}"
  local company="${2:?company required}"
  local role="${3:?role required}"
  local date="${4:-$(date +%Y%m%d)}"
  printf '%s/%s.tex' "$(company_dir "$root" "$company")" "$(resume_basename "$company" "$role" "$date")"
}

resume_pdf_path() {
  local root="${1:?root required}"
  local company="${2:?company required}"
  local role="${3:?role required}"
  local date="${4:-$(date +%Y%m%d)}"
  printf '%s/%s.pdf' "$(company_dir "$root" "$company")" "$(resume_basename "$company" "$role" "$date")"
}
