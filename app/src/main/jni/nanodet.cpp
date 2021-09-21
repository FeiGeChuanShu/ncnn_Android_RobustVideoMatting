// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    is_first = 1;
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    faceseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    faceseg.opt = ncnn::Option();

#if NCNN_VULKAN
    faceseg.opt.use_vulkan_compute = use_gpu;
#endif

    faceseg.opt.num_threads = ncnn::get_big_cpu_count();
    faceseg.opt.blob_allocator = &blob_pool_allocator;
    faceseg.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    faceseg.load_param(parampath);
    faceseg.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    faceseg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    faceseg.opt = ncnn::Option();

#if NCNN_VULKAN
    faceseg.opt.use_vulkan_compute = use_gpu;
#endif

    faceseg.opt.num_threads = ncnn::get_big_cpu_count();
    faceseg.opt.blob_allocator = &blob_pool_allocator;
    faceseg.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    faceseg.load_param(mgr,parampath);
    faceseg.load_model(mgr,modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    if(target_size == 512)
    {
        r1i = ncnn::Mat(128, 128, 16);
        r2i = ncnn::Mat(64, 64, 20);
        r3i = ncnn::Mat(32, 32, 40);
        r4i = ncnn::Mat(16, 16, 64);
    } else{
        r1i = ncnn::Mat(160, 120, 16);
        r2i = ncnn::Mat(80, 60, 20);
        r3i = ncnn::Mat(40, 30, 40);
        r4i = ncnn::Mat(20, 15, 64);
    }
    r1i.fill(0.0f);
    r2i.fill(0.0f);
    r3i.fill(0.0f);
    r4i.fill(0.0f);

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    //TODO:add person detection
    return 0;
}

void NanoDet::matting(cv::Mat &rgb, cv::Mat &mask, cv::Mat &foreground)
{
    ncnn::Extractor ex_face = faceseg.create_extractor();
    ncnn::Mat ncnn_in,ncnn_in1;
    if(target_size == 512)
    {
        ncnn_in = ncnn::Mat::from_pixels_resize(rgb.data,ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows,512,512);
        ncnn_in1 = ncnn::Mat::from_pixels_resize(rgb.data,ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows,256,256);
    }
    else
    {
        ncnn_in = ncnn::Mat::from_pixels_resize(rgb.data,ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows,480,640);
        ncnn_in1 = ncnn::Mat::from_pixels_resize(rgb.data,ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows,240,320);
    }

    const float means[3] = {0,0,0};
    const float norms[3] = {1/255.0,1/255.0,1/255.0};
    ncnn_in.substract_mean_normalize(means, norms);
    ncnn_in1.substract_mean_normalize(mean_vals, norm_vals);
    if (is_first==0)
    {
        r1i = r1o.clone();
        r2i = r2o.clone();
        r3i = r3o.clone();
        r4i = r4o.clone();
    }
    ex_face.input("src1", ncnn_in1);
    ex_face.input("src2", ncnn_in);
    ex_face.input("r1i", r1i);
    ex_face.input("r2i", r2i);
    ex_face.input("r3i", r3i);
    ex_face.input("r4i", r4i);

    ncnn::Mat pha,fgr;
    ex_face.extract("r4o", r4o);
    ex_face.extract("r3o", r3o);
    ex_face.extract("r2o", r2o);
    ex_face.extract("r1o", r1o);
    ex_face.extract("pha",pha);
    ex_face.extract("fgr",fgr);

    float *pha_data = (float*)pha.data;
    float *fgr_data = (float*)fgr.data;

    cv::Mat cv_mask = cv::Mat::zeros(pha.h, pha.w, CV_8UC1);
    cv::Mat cv_pha = cv::Mat(pha.h, pha.w, CV_32FC1, pha_data);

    cv::Mat cv_fgr = cv::Mat::zeros(fgr.h, fgr.w, CV_8UC3);
    cv::Mat cv_foreground = cv::Mat(fgr.h, fgr.w, CV_32FC3);
    if(target_size == 512)
    {
        for (int i = 0; i < 512; i++)
        {
            for (int j = 0; j < 512; j++)
            {
                cv_foreground.at<cv::Vec3f>(i, j)[0] = fgr_data[0 * 512 * 512 + i * 512 + j];
                cv_foreground.at<cv::Vec3f>(i, j)[1] = fgr_data[1 * 512 * 512 + i * 512 + j];
                cv_foreground.at<cv::Vec3f>(i, j)[2] = fgr_data[2 * 512 * 512 + i * 512 + j];
            }
        }
    } else{
        for (int i = 0; i < 640; i++)
        {
            for (int j = 0; j < 480; j++)
            {
                cv_foreground.at<cv::Vec3f>(i, j)[0] = fgr_data[0 * 480 * 640 + i * 480 + j];
                cv_foreground.at<cv::Vec3f>(i, j)[1] = fgr_data[1 * 480 * 640 + i * 480 + j];
                cv_foreground.at<cv::Vec3f>(i, j)[2] = fgr_data[2 * 480 * 640 + i * 480 + j];
            }
        }
    }

    cv_pha.convertTo(cv_mask, CV_8UC1, 255.0, 0);
    cv_foreground.convertTo(cv_fgr, CV_8UC3, 255.0, 0);

    cv_mask.copyTo(mask);
    cv_fgr.copyTo(foreground);
    is_first = 0;
}

int NanoDet::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    cv::Mat mask, fgr;
    matting(rgb,mask,fgr);
    cv::Mat alpha;
    cv::resize(mask, alpha, rgb.size(), 0, 0, 1);
    cv::resize(fgr, fgr, rgb.size(), 0, 0, 1);

    for (int i = 0; i < alpha.rows; i++)
    {
        for (int j = 0; j < alpha.cols; j++)
        {
            uchar data = alpha.at<uchar>(i, j);
            float alpha = (float)data / 255;
            rgb.at < cv::Vec3b>(i, j)[0] = fgr.at < cv::Vec3b>(i, j)[0] * alpha + (1 - alpha) * 120;
            rgb.at < cv::Vec3b>(i, j)[1] = fgr.at < cv::Vec3b>(i, j)[1] * alpha + (1 - alpha) * 255;
            rgb.at < cv::Vec3b>(i, j)[2] = fgr.at < cv::Vec3b>(i, j)[2] * alpha + (1 - alpha) * 155;
        }
    }
    return 0;
}
