
TwoStageDetector.simple_test()
>    x = self.extract_feat(img) # img:[1, 3, 800, 1216] --> x: [[1, 256, 200, 304], [1, 256, 100, 152], [1, 256, 50, 76], [1, 256, 25, 38], [1, 256, 13, 19]]    
>    proposal_list = self.rpn_head.simple_test_rpn(x, img_metas) # ---> proposal_list: [[1000,5]]
>>        rpn_outs = self(x) # --> rpn_cls_score:[[1, 3, 200, 304], [1, 3, 200, 304], [1, 3, 100, 152], [1, 3, 50, 76], [1, 3, 25, 38], [1, 3, 13, 19]], rpn_bbox_pred:[[1, 12, 200, 304], [1, 12, 100, 152], [1, 12, 50, 76], [1, 12, 25, 38], [1, 12, 13, 19]
>>>            multi_apply - RPNHead.forward_single(self, x), for x[0]:
>>>>            x = self.rpn_conv(x) # x: [1, 256, 200, 304] --> x: [1, 256, 200, 304]
>>>>            x = F.relu(x, inplace=True) # x: [1, 256, 200, 304] --> x: [1, 256, 200, 304]
>>>>            rpn_cls_score = self.rpn_cls(x) # x: [1, 256, 200, 304] --> rpn_cls_score: [1, 3, 200, 304] 
>>>>            rpn_bbox_pred = self.rpn_reg(x) # x: [1, 256, 200, 304] --> rpn_bbox_pred: [1, 12, 200, 304]
>>>>            return rpn_cls_score, rpn_bbox_pred
>>        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas) # -->[[1000,5]]
>   return self.roi_head{StandardRoIHead}.simple_test(x, proposal_list, img_metas, rescale=rescale) # --> [[[x1,y1,x2,y2,score]*num_detections_in_class]*80]
>>        det_bboxes, det_labels = self(BBoxTestMixin).simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=rescale) # --> det_bboxes: [[37, 5]], det_labels: [[37]]
>>            rois = bbox2roi(proposals) # [[1000,5((x1, y1, x2, y2, score))]] --> [1000,5([batch_ind, x1, y1, x2, y2])] // Note: the rpn_score is thrown away!
>>            bbox_results = self(StandardRoIHead)._bbox_forward(x, rois) --> cls_score: [1000, 81], bbox_pred: [1000, 320], bbox_feats: [1000, 256, 7, 7]
>>>                bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
>>>>                    target_lvls = self.map_roi_levels(rois, num_levels)
>>>>                    for i in range(num_levels):
>>>>                        mask = target_lvls == i
>>>>                        inds = mask.nonzero(as_tuple=False).squeeze(1)
>>>>                        rois_ = rois[inds]
>>>>                        roi_feats_t = self.roi_layers[i](feats[i], rois_)
>>>>>                            RoIAlign(nn.Module).forward(self, input, rois):
>>>>>                            ext_module(mmcv).roi_align_forward(input, rois, output...)
>>>>                        roi_feats[inds] = roi_feats_t
>>>                cls_score, bbox_pred = self{ConvFCBBoxHead}.bbox_head(bbox_feats)
>>>>                    x = x.flatten(1)
>>>>                    for fc in self.shared_fcs:
>>>>                        x = self.relu(fc(x))
>>>>                    # separate branches
>>>>                    x_cls = x
>>>>                    x_reg = x
>>>>                    cls_score = self.fc_cls(x_cls) 
>>>>                    bbox_pred = self.fc_reg(x_reg) 
>>>            det_bbox, det_label = self.bbox_head.get_bboxes(rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg)
>>        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes) for i in range(len(det_bboxes))]
>>        return bbox_results