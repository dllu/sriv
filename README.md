# simple rust image viewer (sriv)

![train](https://i.dllu.net/2025-04-18-23-55-48_b097e58701685a85.png)

![trams](https://i.dllu.net/2025-04-18-23-53-34_91b104fa4cc7bc4a.png)

Similar to [nsxiv](https://github.com/nsxiv/nsxiv) but:

* gpu-accelerated image viewing
* parallel thumbnail generation
* supports images more than 32768 or 65536 px wide or whatever arcane limit that imlib2 has
* works in wayland natively thanks to nannou using wgpu/winit

Still work in progress.

Mostly vibe coded with AI tbh.

# Configuration

You can put custom keybindings in `~/.config/sriv/bindings.toml`

```
 # Open the current image in the default viewer
"ctrl+o" = "xdg-open {file}"
```
