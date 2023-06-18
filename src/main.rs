use anyhow::Result;
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

/// Generates thumbnails for each image
///
/// # Arguments
///
/// * paths: canonicalized paths
/// * thumb_size: maximum height and width of thumbnail in pixels
fn thumbs(paths: &[PathBuf], thumb_size: u32) -> Result<()> {
    dbg!(paths.len());

    let thumb_dir = Path::new("/home/dllu/.cache/sriv/thumb/");
    let thumb_paths = paths
        .iter()
        .map(|p| {
            let stripped = p.strip_prefix(Path::new("/"))?;
            Ok(thumb_dir.join(stripped))
        })
        .collect::<Result<Vec<PathBuf>>>()?;
    for tp in &thumb_paths {
        fs::create_dir_all(tp.parent().unwrap())?;
    }
    let paths_and_thumbs: Vec<_> = paths.iter().zip(thumb_paths.iter()).collect();
    paths_and_thumbs
        .par_iter()
        .try_for_each(|(p, tp)| -> Result<()> {
            let img = image::open(p)?;
            let small_img = img.thumbnail(thumb_size, thumb_size);
            small_img.save(tp)?;
            Ok(())
        })?;
    Ok(())
}
fn main() {
    let tic = Instant::now();
    let paths = std::env::args()
        .skip(1)
        .map(|x| PathBuf::from(x.as_str()).canonicalize().unwrap())
        .collect::<Vec<PathBuf>>();

    thumbs(&paths, 512_u32).unwrap();
    dbg!(tic.elapsed().as_millis());
}
