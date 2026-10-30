#!/bin/sh
set -eu

if [ "$#" -ne 0 ]; then
    printf 'Usage: %s\n' "$0" >&2
    exit 1
fi

for command in sriv xdg-mime update-desktop-database; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 1
    fi
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
applications_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
desktop_file="$applications_dir/sriv.desktop"

install -Dm644 "$script_dir/../data/sriv.desktop" "$desktop_file"
update-desktop-database "$applications_dir"

sed -n 's/^MimeType=//p' "$desktop_file" | tr ';' '\n' |
    while IFS= read -r mime_type; do
        if [ -n "$mime_type" ]; then
            xdg-mime default sriv.desktop "$mime_type"
        fi
    done

printf 'sriv is now the default viewer for its supported image types.\n'
