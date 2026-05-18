class Metrics:

    def get_iou(self, myDet : list[int], gt : list[int]) -> float:
        my_x1, my_x2, my_y1, my_y2 = myDet
        x1, x2, y1, y2 = gt

        x_left = max(my_x1,x1)
        y_top = max(my_y1, y1)
        x_right = min(my_x2, x2)
        y_bottom = min(my_y2, y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        area_inter = (x_right - x_left) * (y_bottom - y_top)
        area_det = (my_x2 - my_x1) * (my_y2 - my_y1)
        area_gt = (x2 - x1) * (y2 - y1)
        area_union = area_gt + area_det - area_inter
        if area_union == 0:
            return 0.0
        iou = area_inter / area_union
        return iou

    def aux_map(self, detections , ground_truths, iou_threshold = 0.5):
        TP,FN,FP = 0, 0, 0

        duplicates = set()

        for d in detections:
            best_iou = 0
            i_max = 0
            for i,gt in enumerate(ground_truths):
                if i in duplicates:
                    continue
                iou = self.get_iou(d,gt)
                if iou > best_iou:
                    best_iou = iou
                    i_max = i

            if best_iou > iou_threshold:
                TP = TP + 1
                duplicates.add(i_max)
            else : FP = FP + 1

        FN = len(ground_truths) - len(duplicates)

        return {"TP": TP, "FP": FP, "FN": FN}

    def compute_map(self, det_boxes, gt_boxes, iou_threshold):
        precision, recall = 0.0, 0.0
        total_TP, total_FN, total_FP = 0, 0, 0
        matched_gt = set()
        images = set(det_boxes) | set(gt_boxes)

        for img in images:
            dets = det_boxes.get(img,[])
            gts = gt_boxes.get(img, [])

            result = self.aux_map(dets,gts, iou_threshold)

            total_FP += result["FP"]
            total_FN += result["FN"]
            total_TP += result["TP"]
        if (total_TP + total_FP) > 0:
            precision = total_TP / (total_TP + total_FP)
        else:
            precision = 0

        if (total_TP + total_FN) > 0:
           recall = total_TP / (total_TP + total_FN)
        else:
                recall = 0
        print(f"TP: {total_TP} | FP: {total_FP} | FN: {total_FN}")
        print(f"Precision : {precision:.3f}")
        print(f"Recall    : {recall:.3f}")

        return precision, recall
