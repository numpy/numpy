# Script to download Bazel binary directly onto a build machine.

set -e

die() {
  printf >&2 "%s\n" "$1"
  exit 1
}

if [ "$#" -ne 3 ]; then
  die "Usage: ${0} <version> <sha256sum> <destination-file>"
fi

version="$1"
checksum="$2"
dest="$3"

temp_dest="$(mktemp)"

mirror_url="https://mirror.bazel.build/github.com/bazelbuild/bazel/releases/download/
  ${version}/bazel-${version}-linux-x86_64"
github_url="https://github.com/bazelbuild/bazel/releases/download/${version}/bazel-${version}-linux-x86_64"

 for url in "${mirror_url}" "${github_url}"; do
  wget -t 3 -O "${temp_dest}" "${url}" \
    && printf "%s  %s\n" "${checksum}" "${temp_dest}" | shasum -a 256 --check \
  mv "${temp_dest}" "${dest}"
  break
done

[ -f "${dest}" ]
chmod +x "${dest}"
ls -l "${dest}"
