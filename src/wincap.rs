use x11rb::{
    connection::Connection,
    protocol::xproto::{AtomEnum, ConnectionExt, ImageFormat},
};

pub fn find_window_id(name: &str) -> anyhow::Result<u32> {
    let (conn, screen_num) = x11rb::connect(None)?;
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    let client_list = conn.intern_atom(false, b"_NET_CLIENT_LIST")?.reply()?.atom;
    let wm_name = conn.intern_atom(false, b"_NET_WM_NAME")?.reply()?.atom;

    let windows = conn
        .get_property(false, root, client_list, AtomEnum::WINDOW, 0, !0)?
        .reply()?;

    for window in windows.value32().unwrap() {
        let window_name = conn
            .get_property(false, window, wm_name, AtomEnum::ANY, 0, 1024)?
            .reply()?;

        let window_name = String::from_utf8(window_name.value)?;
        dbg!(&window_name);

        if window_name.contains(name) {
            println!("Found window: {} {}", window, window_name);
            return Ok(window);
        }
    }

    conn.flush().unwrap();
    Err(anyhow::anyhow!("Window not found"))
}

pub fn rgba8_to_rgb8(input: image::RgbaImage) -> image::RgbImage {
    let width = input.width() as usize;
    let height = input.height() as usize;

    // Get the raw image data as a vector
    let input: &Vec<u8> = input.as_raw();

    // Allocate a new buffer for the RGB image, 3 bytes per pixel
    let mut output_data = vec![0u8; width * height * 3];

    // Iterate through 4-byte chunks of the image data (RGBA bytes)
    for (output, chunk) in output_data.chunks_exact_mut(3).zip(input.chunks_exact(4)) {
        // ... and copy each of them to output, leaving out the A byte
        output.copy_from_slice(&chunk[0..3]);
    }

    // Construct a new image
    image::ImageBuffer::from_raw(width as u32, height as u32, output_data).unwrap()
}

pub struct WinCap {
    conn: x11rb::rust_connection::RustConnection,
    window_id: u32,
    geometry: x11rb::protocol::xproto::GetGeometryReply,
}

impl WinCap {
    pub fn new(window_id: u32) -> anyhow::Result<Self> {
        let (conn, _) = x11rb::connect(None)?;
        let geometry = conn.get_geometry(window_id)?.reply()?;
        Ok(Self {
            conn,
            window_id,
            geometry,
        })
    }

    pub fn capture(&self) -> anyhow::Result<image::RgbImage> {
        // Get geometry

        let pixmap = self
            .conn
            .get_image(
                ImageFormat::Z_PIXMAP,
                self.window_id,
                0,
                0,
                self.geometry.width,
                self.geometry.height,
                0xffffffff,
            )?
            .reply()?;

        let im = image::RgbaImage::from_vec(
            self.geometry.width as u32,
            self.geometry.height as u32,
            pixmap.data,
        )
        .unwrap();

        Ok(rgba8_to_rgb8(im))
    }
}
