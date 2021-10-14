# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn

class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        metric_dict = dict()
        for k in self.topk:
            metric_dict["top{}".format(k)] = paddle.metric.accuracy(
                x, label, k=k)
        return metric_dict

class mAP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        
        choosen_indices = paddle.argsort(similarities_matrix, axis=1, descending=True) 
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1,0])
        gallery_labels_transpose = paddle.broadcast_to(gallery_labels_transpose, shape=[choosen_indices.shape[0],  gallery_labels_transpose.shape[1]])
        choosen_label = paddle.index_sample(gallery_labels_transpose, choosen_indices)        
        equal_flag = paddle.equal(choosen_label, query_img_id)
        equal_flag = paddle.cast(equal_flag, 'float32')

        acc_sum = paddle.cumsum(equal_flag, axis=1)
        div = paddle.arange(acc_sum.shape[1]).astype("float32") + 1
        precision =  paddle.divide(acc_sum, div)

        #calc map
        precision_mask = paddle.multiply(equal_flag, precision)
        ap = paddle.sum(precision_mask, axis=1) / paddle.sum(equal_flag, axis=1)
        metric_dict["mAP"] = paddle.mean(ap).numpy()[0]
        return metric_dict

class mINP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        
        choosen_indices = paddle.argsort(similarities_matrix, axis=1, descending=True) 
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1,0])
        gallery_labels_transpose = paddle.broadcast_to(gallery_labels_transpose, shape=[choosen_indices.shape[0],  gallery_labels_transpose.shape[1]])
        choosen_label = paddle.index_sample(gallery_labels_transpose, choosen_indices)        
        tmp = paddle.equal(choosen_label, query_img_id)
        tmp = paddle.cast(tmp, 'float64')

        #do accumulative sum
        div = paddle.arange(tmp.shape[1]).astype("float64") + 2
        minus =  paddle.divide(tmp, div)
        auxilary =  paddle.subtract(tmp, minus)
        hard_index = paddle.argmax(auxilary, axis=1).astype("float64")
        all_INP = paddle.divide(paddle.sum(tmp, axis=1), hard_index)
        mINP = paddle.mean(all_INP)
        metric_dict["mINP"] = mINP.numpy()[0]
        return metric_dict

class Recallk(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()

        #get cmc
        choosen_indices = paddle.argsort(similarities_matrix, axis=1, descending=True) 
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1,0])
        gallery_labels_transpose = paddle.broadcast_to(gallery_labels_transpose, shape=[choosen_indices.shape[0],  gallery_labels_transpose.shape[1]])
        choosen_label = paddle.index_sample(gallery_labels_transpose, choosen_indices)        
        equal_flag = paddle.equal(choosen_label, query_img_id)
        equal_flag = paddle.cast(equal_flag, 'float32')
        
        acc_sum = paddle.cumsum(equal_flag, axis=1)
        mask = paddle.greater_than(acc_sum, paddle.to_tensor(0.)).astype("float32")
        all_cmc = paddle.mean(mask, axis=0).numpy() 

        for k in self.topk:
            metric_dict["recall{}".format(k)] = all_cmc[k - 1]
        return metric_dict

class DistillationTopkAcc(TopkAcc):
    def __init__(self, model_key, feature_key=None, topk=(1, 5)):
        super().__init__(topk=topk)
        self.model_key = model_key
        self.feature_key = feature_key

    def forward(self, x, label):
        x = x[self.model_key]
        if self.feature_key is not None:
            x = x[self.feature_key]
        return super().forward(x, label)
