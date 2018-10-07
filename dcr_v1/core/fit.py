# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import mxnet as mx
import logging
import os
import time

def _save_model(prefix=None, rank=0):
    if prefix is None:
        return None
    dst_dir = os.path.dirname(prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(prefix if rank == 0 else "%s-%d" % (
        prefix, rank))