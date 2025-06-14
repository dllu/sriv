# simple rust image viewer (sriv)

![train](https://i.dllu.net/2025-04-18-23-55-48_b097e58701685a85.png)

![trams](https://i.dllu.net/2025-04-18-23-53-34_91b104fa4cc7bc4a.png)

similar to [nsxiv](https://github.com/nsxiv/nsxiv) but:

* gpu-accelerated image viewing
* parallel thumbnail generation
* supports images more than 32768 or 65536 px wide or whatever arcane limit that imlib2 has
* works in wayland natively thanks to nannou using wgpu/winit

built on [nannou](https://nannou.cc/).

still work in progress.

mostly vibe coded with AI tbh.

# installation

To build and install the program system-wide on Linux, use one of the following methods:

```bash
# Build a release and install the binary to /usr/local/bin
cargo build --release
sudo install -Dm755 target/release/sriv /usr/local/bin/sriv
```

Alternatively, install directly with Cargo:

```bash
sudo cargo install --path . --force --root /usr/local
```

## Usage

To clear and regenerate the thumbnail cache for all specified images, use the `--clear-cache` flag before the file or directory arguments:

```bash
sriv-rs --clear-cache <image files or directories>
```

# configuration

# configuration

you can put custom keybindings in `~/.config/sriv/bindings.toml`. Just put whatever modifiers (`ctrl`, `shift`, `alt`) if you want and `+` and then the letter or number of the key.

```
 # Open the current image in the default viewer
"ctrl+o" = "xdg-open {file}"
```

# faq

> why does it use so much cpu?

it is designed to aggressively generate thumbnails with many threads.

> why does it use so much ram?

in addition to generating thumbnails in parallel, it also stores a local cache of full size images.

> why does it use so much gpu?

it puts the textures on the gpu for smoother viewing experience.
