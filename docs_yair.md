# results = model(return_loss=False, rescale=True, **data):
mmdet.models.detectors.faster_rcnn.FasterRCNN(
>    mmdet.models.detectors.TwoStageDetector(
>>        mmdet.models.detectors.BaseDetector))
            TwoStageDetector.simple_test()
                x = self.extract_feat(img) # backbone (and optional neck)
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
                self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

## proposal_list = self.rpn_head.simple_test_rpn(x, img_metas):
mmdet.models.dense_heads.rpn_head.RPNHead(
>    mmdet.models.dense_heads.anchor_head.AnchorHead(
>>        mmdet.models.dense_heads.base_dense_head.BaseDenseHead(mmcv.runner.BaseModule(nn.module), metaclass=ABCMeta))
>>        mmdet.models.dense_heads.dense_test_mixins.BBoxTestMixin(object))
            BBoxTestMixin.simple_test_rpn()
                rpn_outs = self(x)
                proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)

### rpn_outs = self(x)
        RPNHead.forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

## self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
StandardRoIHead(
    BaseRoIHead(BaseModule, metaclass=ABCMeta),
    BBoxTestMixin,
    mmdet.models.test_mixins.MaskTestMixin)

    StandardRoIHead.simple_test():
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes) for i in range(len(det_bboxes))
        ]

### BBoxTestMixin.simple_test_bboxes():
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        det_bbox, det_label = self.bbox_head.get_bboxes()

#### StandardRoIHead._bbox_forward(x, rois)
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

##### bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois):
SingleRoIExtractor(
    BaseRoIExtractor(BaseModule, metaclass=ABCMeta))

        SingleRoIExtractor.forward(self, feats, rois, roi_scale_factor=None):
            target_lvls = self.map_roi_levels(rois, num_levels)
            for i in range(num_levels):
                mask = target_lvls == i
                inds = mask.nonzero(as_tuple=False).squeeze(1)
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t

###### roi_feats_t = self.roi_layers[i](feats[i], rois_i)
    RoIAlign(nn.Module).forward(self, input, rois):
        ext_module(mmcv).roi_align_forward(input, rois, output...)

###### cls_score, bbox_pred = self.bbox_head(bbox_feats)
mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.ConvFCBBoxHead(BBoxHead).forward(self, x)
    x = x.flatten(1)
    for fc in self.shared_fcs:
        x = self.relu(fc(x))
    # separate branches
    x_cls = x
    x_reg = x
    cls_score = self.fc_cls(x_cls) 
    bbox_pred = self.fc_reg(x_reg) 

#### det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)