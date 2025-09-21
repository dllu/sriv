# simple rust image viewer (sriv)

https://github.com/user-attachments/assets/734e3a02-e9ff-4f24-9c51-27585d53a806

![screenshot](https://i.dllu.net/20250921_13h39m33s_grim_e1a14e98eb3dddf3.png)

* minimalistic UI with vim-like keybindings
* gpu-accelerated image viewing
* parallel thumbnail generation
* supports images more than 32768 or 65536 px wide or whatever arcane limit that imlib2 has
* CLIP-powered semantic search across your library
* works in wayland natively thanks to nannou using wgpu/winit

built on [nannou](https://nannou.cc/).
inspired by [nsxiv](https://github.com/nsxiv/nsxiv).

still work in progress.

mostly vibe coded with AI tbh.

# build and installation

to build and install the program system-wide on Linux, use one of the following methods:

```bash
cargo build --release
sudo install -Dm755 target/release/sriv /usr/local/bin/sriv
```

if you have a CUDA-capable GPU, you can use

```bash
cargo build --release --features=cuda
```

alternatively, install directly with Cargo:

```bash
sudo cargo install --path . --force --root /usr/local
```

## usage

to clear and regenerate the thumbnail cache for all specified images, use the `--clear-cache` flag before the file or directory arguments:

```bash
sriv-rs --clear-cache <image files or directories>
```

### clip semantic search

sriv can index your images with [OpenAI CLIP (ViT-B/32)](https://github.com/openai/CLIP) via the [Hugging Face Candle](https://github.com/huggingface/candle) runtime.
The first launch downloads model weights and tokenizer from the Hugging Face Hub, after which embeddings are cached alongside your thumbnails in `${XDG_CACHE_HOME}/sriv/`.

- press `/` to focus the search bar and type a natural-language prompt. The bar glows purple when focused.
- hit `Enter` to run the search; results are ranked by cosine similarity and highlighted at the top.
- while unfocused in thumbnail mode, `n`/`Shift+n` (or `p`/`Shift+p`) step through the match list, keeping
  search results intact.
- press `/` again to refocus and refine the query, or `Esc`/`Backspace` on an empty field to clear the search.

If built with CUDA support, embedding generation automatically uses CUDA when available; otherwise sriv fans out across your CPU cores.
The status area shows how many embeddings are still pending and whether the GPU or CPU is in use.

# configuration

you can put custom keybindings in `~/.config/sriv/bindings.toml` to execute custom commands.
Just put whatever modifiers (`ctrl`, `shift`, `alt`) if you want and `+` and then the letter or number of the key.

```
 # Open the current image in the default viewer
"ctrl+o" = "xdg-open {file}"
```

# faq

> why does it use so much cpu?

it is designed to aggressively generate thumbnails with many threads

> why does it use so much ram?

in addition to generating thumbnails in parallel, it also stores a local cache of full size images

> why does it use so much gpu?

it puts the textures on the gpu for a smoother viewing experience, and it may use a CUDA-capable GPU for CLIP embedding generation
