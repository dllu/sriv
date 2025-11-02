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
# open the current image in the default viewer
"ctrl+o" = "xdg-open {file}"

# copy the current image to the clipboard
"ctrl+c" = "xclip -selection clipboard -target image/png -i {file}"

# print out the EXIF metadata of the current image
"ctrl+e" = "exiv2 {file}"
```

# design

The goal is to have a super fast, responsive image viewer that can handle tens of thousands of 100 megapixel photos and generate thumbnails/CLIP embeddings in parallel.

* there's a global queue of pending thumbnails, that gets sorted based on distance from viewport every time you scroll.
* *thumb workers*: a bunch of workers each pop off the top of the queue to try to load cached thumbnails/CLIP embeddings in order. If a cached thumbnail isn't available, it generates the thumbnail from the full size image. Loading a cached thumbnail on an SSD takes perhaps a millisecond and loading a full size image to generate a thumbnail can take over a second; regardless, locking the queue to pop off the top is a negligible amount of time and does not lead to significant resource contention even with 16 threads. If CLIP embedding isn't available, it sends the thumbnail through the *pending clip* channel to the CLIP generation workers. It also sends the thumbnail and CLIP embedding (if available) through the *thumb channel* back to the main thread.
* *clip workers*: a bunch of workers generate CLIP embeddings by listening on the *pending clip* channel and generate and cache CLIP embeddings. Once a CLIP embedding is computed, it sends it back to the main thread via the *clip channel*, where the main thread populates the thumbnail store.
* *full size image worker*: Fetches full size images and caches them in the background into an LRU cache. When in thumbnail mode, it tries to fetch the full size version of the selected thumbnail, so that when you do decide to zoom in, you don't have to wait another second for it to load. Also, it fetches the next and previous images when in full size image mode.

All thumbnails are loaded into memory. This is different from `sxiv`/`nsxiv`, which conserve computational resources by only generating thumbnails for images visible in the viewport, but lead to a laggy user experience when viewing lots of high resolution images, where you have to wait for thumbnails to be generated, one per second, whenever you scroll.

However, only the thumbnails visible in the viewport are loaded into textures on the GPU. This is to conserve VRAM, and uploading a small handful of visible textures onto the GPU is very fast.

Also, a tiled texture strategy is used for displaying the full size image, to prevent crashes on certain GPUs or difficulty with allocating a contiguous giant texture.
Full size images can be quite large and use a lot of memory, so an LRU cache keeps memory usage bounded.

# faq

> why does it use so much cpu?

it is designed to aggressively generate thumbnails with many threads

> why does it use so much ram?

in addition to generating thumbnails in parallel, it also stores a local cache of full size images

> why does it use so much gpu?

it puts the textures on the gpu for a smoother viewing experience, and it may use a CUDA-capable GPU for CLIP embedding generation
