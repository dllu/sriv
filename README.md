# simple rust image viewer (sriv)

https://github.com/user-attachments/assets/734e3a02-e9ff-4f24-9c51-27585d53a806

![screenshot](https://i.dllu.net/20250921_13h39m33s_grim_e1a14e98eb3dddf3.png)

* minimalistic UI with vim-like keybindings
* gpu-accelerated image viewing
* animated GIF, WebP, APNG, and JPEG XL playback with per-frame timing
* color-managed ICC, HEIF/NCLX, and JPEG XL images, with Display P3 output when the surface supports it
* parallel thumbnail generation
* supports images more than 32768 or 65536 px wide or whatever arcane limit that imlib2 has
* SigLIP 2 semantic search across your library
* works in Wayland natively with winit and wgpu

built directly on [winit](https://github.com/rust-windowing/winit) and [wgpu](https://wgpu.rs/).
inspired by [nsxiv](https://github.com/nsxiv/nsxiv).

still work in progress.

mostly vibe coded with AI tbh.

# build and installation

to build and install the program, use one of the following methods.

On macOS, HEIF/HEIC decoding uses the system ImageIO framework, so no separate
HEIF library is required.

On Linux, HEIF/HEIC support requires the libheif development package, for example:

```bash
sudo apt install libheif-dev
```

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

On Linux, after installing `sriv` on your `PATH`, run this from the repository
to make it the default image viewer for your user (without `sudo`):

```bash
./scripts/set-default-viewer.sh
```

This installs `data/sriv.desktop` into `${XDG_DATA_HOME:-~/.local/share}/applications`
and uses `xdg-mime` to make sriv the default for JPEG, PNG/APNG, GIF, BMP, TIFF,
WebP, JPEG XL, and HEIF/HEIC images. It requires `xdg-mime` (from `xdg-utils`)
and `update-desktop-database` (from `desktop-file-utils`). File managers and
`xdg-open` will then open supported images in sriv.

## Wayland regression test

Run the surface lifecycle test in a disposable headless Sway session:

```bash
RUSTFLAGS="-C target-cpu=native" python3 scripts/test-wayland.py
```

Sway and `swaymsg` must be installed together; use `--sway /path/to/sway` for a
custom installation. The test disables Xwayland, checks native Wayland window
creation, and exercises live suboptimal frames, zero sizes, surface recovery,
fullscreen, resizing, and moves between outputs with different scales. It uses
the graphics driver and does not change your desktop session. Ordinary
`cargo test` skips this compositor-dependent test.

## usage

In single-image mode, press `=` for one image pixel per physical screen pixel,
or `w` to fit the image to the window. The zoom indicator uses this physical
pixel ratio, independently of display scaling. Thumbnails also keep their cached
size of up to 256 physical pixels per edge, so high-DPI displays show more
thumbnails rather than enlarging them. Text, gaps, and UI controls follow the
display's DPI setting. Existing thumbnail caches can be reused.

to clear and regenerate the thumbnail cache for all specified images, use the `--clear-cache` flag before the file or directory arguments:

```bash
sriv --clear-cache <image files or directories>
```

Use `--private` to generate thumbnails and SigLIP 2 image embeddings only in memory for the current session. This skips reading and writing sriv's disk cache; it does not delete files cached by earlier runs. Combine it with `--clear-cache` to remove those existing files. SigLIP 2 model weights and tokenizer are still managed by the Hugging Face cache.

```bash
sriv --private <image files or directories>
```

### semantic search

sriv indexes your images with [Google SigLIP 2 Base at 512 pixels](https://huggingface.co/google/siglip2-base-patch16-512) via the [Hugging Face Candle](https://github.com/huggingface/candle) runtime. The model revision and preprocessing are pinned; its 768-dimensional embeddings are normalized for cosine similarity.
The first launch downloads approximately 1.5 GB of model weights plus the tokenizer from the Hugging Face Hub. Embeddings and 512-pixel PNG thumbnails are cached in `${XDG_CACHE_HOME}/sriv/`. Thumbnail cache names include their size and color space; embedding cache names include the model revision and preprocessing version. Older caches are ignored and regenerated without deleting them.

- press `/` to focus the search bar and type a natural-language prompt. The bar glows purple when focused.
- hit `Enter` to run the search; results are ranked by cosine similarity and highlighted at the top.
- while unfocused in thumbnail mode, `n`/`Shift+n` (or `p`/`Shift+p`) step through the match list, keeping
  search results intact.
- press `/` again to refocus and refine the query, or `Esc`/`Backspace` on an empty field to clear the search.

If built with CUDA support, embedding generation uses half-precision CUDA inference when available; otherwise sriv fans out across your CPU cores. CPU workers share one set of model weights. Text queries take priority between image batches, so indexing does not put a query behind the entire image queue.
The status area shows how many embeddings are still pending and whether the GPU or CPU is in use.

# configuration

you can put custom keybindings in `~/.config/sriv/bindings.toml` to execute custom commands.
Just put whatever modifiers (`ctrl`, `shift`, `alt`) if you want and `+` and then the letter or number of the key.

`{file}` is automatically shell-quoted, so paths containing spaces or special characters
are passed as one argument. Leave `{file}` unquoted in your commands.

```
# open the current image in the default viewer
"ctrl+o" = "xdg-open {file}"

# copy the current image to the clipboard
"ctrl+c" = "xclip -selection clipboard -target image/png -i {file}"

# print out the EXIF metadata of the current image
"ctrl+e" = "exiv2 {file}"
```

if you want a binding to launch a graphical app and not attach it to sriv's terminal panel, use the table form and set `terminal = false`.

```toml
["ctrl+d"]
command = "~/proj/pupphoto/open_darktable.sh {file}"
terminal = false
```

you can also put general UI settings in `~/.config/sriv/config.toml`.

```toml
# optional: path to a .ttf or .otf font to use for all UI text
ui_font_path = "/usr/share/fonts/noto/NotoSansMono-Regular.ttf"
```

if `ui_font_path` is unset or fails to load, sriv falls back to the system sans-serif font.

# design

The goal is to have a super fast, responsive image viewer that can handle tens of thousands of 100 megapixel photos and generate thumbnails and SigLIP 2 embeddings in parallel.

The window opens before the library is scanned. A background worker lists and sorts
candidate paths, checks their contents, and streams small batches to the thumbnail
queue. Thumbnails appear while discovery continues; the image count shows when the
scan is still in progress. Modification times arrive with each batch, and XMP
sidecars are scanned afterward. Slow source-file reads never block the first frame.

* there's a global queue of pending thumbnails, that gets sorted based on distance from viewport every time you scroll.
* *thumb workers*: load cached PNG bytes, or decode the original image once to generate a 512-pixel thumbnail. Send the compressed PNG and any cached embedding to the event loop. If an embedding is missing, send a shared reference to the same compressed bytes to the inference queue. Queued inference never retains a second library of raw pixels.
* *thumbnail decode workers*: decode PNG bytes from RAM for the viewport and nearby rows. Scrolling replaces queued work with the latest viewport. A 128 MiB decoded-image LRU keeps recently viewed thumbnails available; GPU textures are retained only around the viewport. No thumbnail decompression runs on the event loop.
* *search workers*: decode queued PNGs, convert their display color space to sRGB, resize the complete image to 512×512, and generate SigLIP 2 embeddings. CPU workers process one image at a time; CUDA batches up to four. Queries take priority between image batches. Versioned results return to the event loop.
* *full size image worker*: Fetches full size images and caches them in the background into an LRU cache. When in thumbnail mode, it tries to fetch the full size version of the selected thumbnail, so that when you do decide to zoom in, you don't have to wait another second for it to load. Also, it fetches the next and previous images when in full size image mode.

All compressed thumbnails are loaded into RAM, so scrolling through a warmed library requires no thumbnail disk reads. Only thumbnails near the viewport and a small decoded LRU retain raw pixels. PNG compression is lossless, and cache generation encodes the PNG only once for both the disk and memory copies. `--private` keeps the compressed cache in RAM without writing it to disk.

Also, a tiled texture strategy is used for displaying the full size image, to prevent crashes on certain GPUs or difficulty with allocating a contiguous giant texture.
Full size images can be quite large and use a lot of memory, so an LRU cache keeps memory usage bounded.

## state machines

The app has several independent lifecycles. Keep independent concerns such as search
focus and execution separate; use enums when states have different data or when
several flags would allow contradictory combinations. The event loop owns UI state;
workers send versioned results and never mutate it directly.

| Lifecycle | States and transitions | Owner and invariant |
| --- | --- | --- |
| Application | Uninitialized → running → exiting; startup failure → exit | [`SrivApplication`](src/app.rs). Repeated resume events do not create another window or model. |
| Library discovery | `Discovering` → batches of images and sidecars → `Ready` | [`LibraryState`](src/library.rs). Directory enumeration, content detection, initial modification times, cache clearing, and sidecar scans run off the event loop. Append stable indices and publish worker paths before queueing each batch; bound pending batches and yield to input and drawing. |
| Worker wakeups | Idle → wake queued → drain channels → idle | [`AppProxy`](src/app.rs). Clear the wake flag before draining so work arriving during the drain can schedule another wake. |
| Event scheduling | Waiting → deadline reached or worker/input event → update → waiting | [`ApplicationState`](src/app.rs). Wait for the earliest file poll, selection preload, load retry, or animation frame. Input redraws are coalesced. |
| Physical keys | Released → pressed while focused → repeating → released; held across focus → ignored until release | [`Keys`](src/app.rs). Losing focus resets modifiers; synthetic presses and unfocused repeats cannot trigger shortcuts. |
| Input routing | Search editing, terminal controls, viewer controls, custom bindings | [`window_event` and `key_pressed`](src/app.rs). Search gets first refusal; the event that opens search contributes no text. Custom bindings can still run after viewer actions. |
| Viewing and zoom | Thumbnails ↔ single image; fitted ↔ manual zoom; windowed ↔ fullscreen | [`Mode`](src/state.rs) and navigation in [`app.rs`](src/app.rs). Navigating starts fitted even when decoding is pending. Manual zoom survives resizing until fit is requested again. |
| Selection preload | `Waiting { deadline }` → `Requested`; thumbnail input or selected-file invalidation rearms the delay | [`SelectionPreload`](src/state/loading.rs). A stable selection schedules one preload after 200 ms. |
| File watching | Observed revision → changed/deleted/reappeared → invalidate → reload | [`check_image_modification`](src/app.rs), [`ImageStates`](src/state/loading.rs). Increment the generation and invalidate thumbnails, embeddings, search scores, and full images together. Poll the current/visible files plus a rotating batch. |
| Thumbnail queue | Open/empty ↔ queued → dispatched; open → closed/draining → exhausted | [`ThumbRequestQueue`](src/state.rs). One queued entry per index carries its latest generation; closing wakes waiters and rejects new work. |
| Thumbnail residency | Requested → compressed PNG in RAM → decode queued/in flight → decoded LRU → visible GPU texture | [`ThumbnailStore` and `ThumbnailDecoder`](src/thumbnails.rs). Scrolling replaces queued decode work. The decoded LRU is bounded by bytes; leaving the viewport releases GPU textures. Results must match the current file generation. Corrupt disk PNGs are removed by workers and regenerated. |
| Disk caches | Missing/stale/unreadable → generate → optional write; valid → reuse | [`image_io.rs`](src/image_io.rs), [`clip.rs`](src/clip.rs). Existing mtime checks govern cache freshness. Private mode skips sriv cache reads/writes. A failed embedding-cache write does not discard a usable in-memory embedding. |
| Search workers | Initialize model/tokenizer → CPU or GPU inference → wait/work → stopped | [`ClipEngine`](src/clip.rs). CUDA initialization can fall back to CPU. When all workers stop, pending work becomes a visible failure; thumbnail decoding and cached embeddings remain usable. |
| Search job batching | Receive prioritized text or image → infer text or image batch → receive again | [`run_worker`](src/clip.rs). Text queries use a separate priority queue and run between image batches. Worker results may arrive out of order. |
| Per-image embeddings | `WaitingForThumbnail` → `Pending` → `Ready` or `Failed`; cached thumbnail → `Ready`; invalidation → waiting | [`ImageStates` and `EmbeddingState`](src/state/loading.rs). An early embedding completion cannot be undone by later thumbnail delivery. Progress counts update with transitions, so rendering does not scan the library. |
| Search | Absent ↔ present; focused ↔ unfocused; `Draft` → `Pending` → `Ready` or `Failed`; edit → draft | [`SearchState`](src/state/search.rs). Only the pending request ID can complete. Results and their query embedding exist together. Live indexing preserves the selected image when rankings change; invalidation removes stale scores and preserves the navigation position until the selected image reloads or the user advances. |
| Full-image loading | Absent → `InFlight` → `Ready` or `Failed`; failure deadline → retry; invalidation → absent | [`FullImageCache`](src/state/loading.rs). A result must match the outstanding generation. The five-second retry is eligible exactly at its deadline; errors are displayed instead of indefinite loading. |
| Full-image residency | Cached → most/least recently used → evicted → absent | [`FullImageCache`](src/state/loading.rs). Load state, textures, and recency share one owner. At most four images are resident; late preloads cannot evict the selected image. |
| Full-image GPU tiles | Raw pixels → texture uploaded on demand → released with eviction | [`Tile`](src/state.rs), [`prepare_current_full_textures`](src/ui.rs). Only prepared tiles are drawn; unsupported 16-bit formats use the existing fallback conversion. |
| Playback | Unscheduled → restart on first frame → frame deadlines → next frame/cycle | [`TiledTexture`](src/state.rs). Still images have no deadline. Frame delays are clamped to 10 ms and catch-up after suspension traverses at most one cycle. Only single-image viewing schedules playback. |
| GPU surface | Zero-sized/skipped → needs configuration → ready → frame acquired → presented; suboptimal/outdated/lost → deferred recovery | [`Renderer`](src/renderer.rs). Never configure a zero-sized surface or reconfigure while a live frame is owned. Timeout requests another draw; occlusion waits for window events. |
| Terminal panel | Hidden ↔ visible; empty ↔ sessions with active tab; following output ↔ scrollback | [`TerminalState`](src/state.rs), [`ui.rs`](src/ui.rs). Active tabs are clamped after removal; closing the last tab hides the panel. Output is routed by session ID, so closed tabs ignore late events. |
| Terminal commands | `Running` → `Exited`; startup/read/wait failure → `Failed`; PTY output closes independently | [`CommandStatus`](src/state.rs), [`start_command_workers`](src/ui.rs). Report process exit immediately and retain the master until output is drained, even if a descendant keeps the PTY open. Keep the first failure even if a later exit arrives. |
| Detached commands | Spawn → run outside the panel → wait/reap | [`launch_detached_command`](src/ui.rs). Waiting happens on a background thread and leaves the UI responsive. |

When changing a lifecycle, put its transition in its owning type and test event
orderings, cancellation/invalidation, failure recovery, and deadline boundaries.
Do not add independent flags for data that belongs to one phase. Simple binary
states and state derived from window/channel APIs do not need a general FSM framework.

Run the transition and image-format tests with `cargo test --locked`. On ARM hosts
where Candle's GEMM dependency requires FP16 target features, use
`RUSTFLAGS="-C target-cpu=native" cargo test --locked`. The separate Wayland test
above exercises the surface lifecycle with real GPU frames and output-scale changes.

# faq

> why does it use so much cpu?

it is designed to aggressively generate thumbnails with many threads

> why does it use so much ram?

in addition to generating thumbnails in parallel, it also stores a local cache of full size images

> why does it use so much gpu?

it puts the textures on the gpu for a smoother viewing experience, and it may use a CUDA-capable GPU for SigLIP 2 embedding generation
