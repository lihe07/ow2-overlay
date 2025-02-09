use std::cmp::Ordering;

use crate::BBOX;

// Internal representation with precomputed corners and area.
struct BBoxInternal {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    conf: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    area: f32,
    suppressed: bool,
}

pub fn nms_center_opt(mut bboxes: Vec<BBOX>, iou_threshold: f32) -> Vec<BBOX> {
    // Convert each center-based bbox to one with its corner coordinates and area.
    let mut boxes: Vec<BBoxInternal> = bboxes
        .into_iter()
        .map(|(cx, cy, w, h, conf)| {
            let half_w = w * 0.5;
            let half_h = h * 0.5;
            let x1 = cx - half_w;
            let y1 = cy - half_h;
            let x2 = cx + half_w;
            let y2 = cy + half_h;
            let area = (x2 - x1) * (y2 - y1);
            BBoxInternal {
                cx,
                cy,
                w,
                h,
                conf,
                x1,
                y1,
                x2,
                y2,
                area,
                suppressed: false,
            }
        })
        .collect();

    // Sort boxes in descending order by confidence.
    boxes.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(Ordering::Equal));

    let mut retained = Vec::new();
    let len = boxes.len();

    for i in 0..len {
        if boxes[i].suppressed {
            continue;
        }
        // Retain the current box.
        let current = &boxes[i];
        retained.push((current.cx, current.cy, current.w, current.h, current.conf));

        // Suppress all boxes that overlap too much with the current one.
        for j in (i + 1)..len {
            if boxes[j].suppressed {
                continue;
            }
            let inter_x1 = boxes[i].x1.max(boxes[j].x1);
            let inter_y1 = boxes[i].y1.max(boxes[j].y1);
            let inter_x2 = boxes[i].x2.min(boxes[j].x2);
            let inter_y2 = boxes[i].y2.min(boxes[j].y2);
            let inter_area = if inter_x2 > inter_x1 && inter_y2 > inter_y1 {
                (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            } else {
                0.0
            };
            let union = boxes[i].area + boxes[j].area - inter_area;
            if union > 0.0 && (inter_area / union) > iou_threshold {
                boxes[j].suppressed = true;
            }
        }
    }

    retained
}
