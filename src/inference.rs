use std::{cmp::Ordering, path::Path};

use image::DynamicImage;
use ort::{
    execution_providers, inputs,
    session::{builder::GraphOptimizationLevel, Session},
};

use crate::BBOX;

fn center_to_corners(b: BBOX) -> BBOX {
    let (cx, cy, w, h, s) = b;
    let half_w = w * 0.5;
    let half_h = h * 0.5;
    let x1 = cx - half_w;
    let y1 = cy - half_h;
    let x2 = cx + half_w;
    let y2 = cy + half_h;
    (x1, y1, x2, y2, s)
}

fn iou_center(a: BBOX, b: BBOX) -> f32 {
    // Convert center-based (cx, cy, w, h) to corner-based (x1, y1, x2, y2)
    let (ax1, ay1, ax2, ay2, _) = center_to_corners(a);
    let (bx1, by1, bx2, by2, _) = center_to_corners(b);

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_area = if inter_x2 > inter_x1 && inter_y2 > inter_y1 {
        (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    } else {
        0.0
    };

    let area_a = (ax2 - ax1) * (ay2 - ay1);
    let area_b = (bx2 - bx1) * (by2 - by1);

    inter_area / (area_a + area_b - inter_area)
}

pub fn nms_center(mut bboxes: Vec<BBOX>, iou_threshold: f32) -> Vec<BBOX> {
    // Create indices and sort them based on confidence descending
    let mut indices: Vec<usize> = (0..bboxes.len()).collect();
    indices.sort_by(|&i, &j| {
        bboxes[j]
            .4
            .partial_cmp(&bboxes[i].4)
            .unwrap_or(Ordering::Equal)
    });

    let mut retained_boxes = Vec::new();

    // Iterate in order of descending confidence, picking the highest boxes first
    while let Some(current_idx) = indices.pop() {
        // If already suppressed, skip
        if bboxes[current_idx].4 < 0.0 {
            continue;
        }

        // Retain this box
        let current_box = bboxes[current_idx];
        retained_boxes.push(current_box);

        // Suppress boxes that have high overlap with the currently selected box
        for &other_idx in indices.iter() {
            if bboxes[other_idx].4 >= 0.0 {
                let overlap = iou_center(current_box, bboxes[other_idx]);
                if overlap > iou_threshold {
                    bboxes[other_idx].4 = -1.0; // Mark as suppressed
                }
            }
        }
    }

    retained_boxes
}

pub struct Model {
    sess: ort::session::Session,
    resizer: fast_image_resize::Resizer,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl Model {
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> anyhow::Result<Self> {
        ort::init_from("./onnxruntime-linux-x64-gpu-1.20.1/lib/libonnxruntime.so").commit()?;

        let sess = Session::builder()?
            .with_execution_providers([
                execution_providers::TensorRTExecutionProvider::default()
                    .with_engine_cache(true)
                    .with_engine_cache_path("./engine_cache")
                    .with_builder_optimization_level(5)
                    .with_fp16(true)
                    .build(),
                // execution_providers::CUDAExecutionProvider::default().build(),
                // execution_providers::CPUExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;

        let resizer = fast_image_resize::Resizer::new();

        Ok(Self {
            sess,
            resizer,
            conf_threshold,
            iou_threshold,
        })
    }

    pub fn process_img(&mut self, im: image::RgbImage) -> anyhow::Result<Vec<crate::BBOX>> {
        // preprocess: resize to 640x640

        let mut dest_im = DynamicImage::ImageRgb8(image::RgbImage::new(640, 640));

        self.resizer
            .resize(&DynamicImage::ImageRgb8(im), &mut dest_im, None)?;

        let dest_im = dest_im.into_rgb8();

        // Convert to ndarray
        let im = ndarray::Array3::from_shape_vec((640, 640, 3), dest_im.into_raw());
        // Convert to float and normalize
        let im = im.unwrap().mapv(|x| x as f32 / 255.0);
        // Transpose to (3, 640, 640)
        let im = im.permuted_axes([2, 0, 1]);
        // Add batch dimension
        let im = im.insert_axis(ndarray::Axis(0));

        let mut outputs = self.sess.run(inputs!["images" => im]?)?;
        let output = outputs.get_mut("output0").unwrap();
        // Postprocess

        let pred = output.try_extract_tensor_mut::<f32>()?;
        let pred = pred.remove_axis(ndarray::Axis(0));
        let pred = pred.into_dimensionality::<ndarray::Ix2>()?;
        let pred = pred.permuted_axes([1, 0]);

        // pred: x, y, w, h, confidence, junk
        let pred = pred
            .outer_iter()
            .filter(|x| x[4] > self.conf_threshold)
            .map(|x| (x[0], x[1], x[2], x[3], x[4]))
            .collect::<Vec<_>>();

        Ok(crate::nms::nms_center_opt(pred, self.iou_threshold)
            .into_iter()
            .map(|x| (x.0 / 640.0, x.1 / 640.0, x.2 / 640.0, x.3 / 640.0, x.4))
            .collect())
    }
}
