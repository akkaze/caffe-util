#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <random>
#include <vector>

#include <time.h>
#include <string.h>

#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/threadpool.hpp>
#include <boost/thread.hpp>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/format.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define USE_LEVELDB
#ifdef USE_LEVELDB
#include "leveldb/db.h"
#endif
#include "lmdb.h"
namespace fs = boost::filesystem;
namespace treadpool = boost::threadpool;
#define PATCH_SIZE 9
#define HALF_PATCH_SIZE 4
#define PIXEL_COUNT 81

#define NEG_LOW 6
#define NEG_HIGH 10

#define POS 0
bool is_inside(const cv::Point2i& pt,const cv::Mat& img)
{
    return (pt.x >= 0 && pt.x < img.cols &&
            pt.y >= 0 && pt.y< img.rows);
}

const cv::Mat crop_image_patch(const cv::Mat& img,const int& cx,const int& cy)
{
    assert(!img.empty());
    cv::Mat patch;

    cv::Point2i tl(cx - HALF_PATCH_SIZE,
                   cy - HALF_PATCH_SIZE);
    cv::Point2i rb(cx + HALF_PATCH_SIZE,
                   cy + HALF_PATCH_SIZE);

    if(!is_inside(tl,img) || !is_inside(rb,img))
        return patch;
    else
    {
        cv::Rect rect(cx - HALF_PATCH_SIZE,
                      cy - HALF_PATCH_SIZE,
                      PATCH_SIZE,
                      PATCH_SIZE);
        img(rect).copyTo(patch);
        assert(!patch.empty() && (patch.rows == PATCH_SIZE)
               && (patch.cols == PATCH_SIZE));
        return patch;
    }
}

std::vector<std::string> list_filename(const char* dirname)
{
    std::vector<std::string> filenames;
    fs::path dir_path(dirname);
    fs::directory_iterator end_itr;
    for(fs::directory_iterator itr(dir_path); itr != end_itr; ++itr)
    {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (is_regular_file(itr->path())) {
            // assign current file name to current_file and echo it out to the console.
            const std::string& current_file = itr->path().string();
            filenames.push_back(current_file);
        }
    }
    return filenames;
}

class Random {
public:
    Random();
    Random(std::mt19937::result_type _seed,int _min,int _max)
    {
        std::uniform_int_distribution<>::param_type param(_min,_max);
        eng.seed(_seed);
        dist.param(param);
    }

    int draw_number()
    {
        return dist(eng);
    }
private:
    std::mt19937 eng;
    std::uniform_int_distribution<> dist;
};


int generate_negtive_offset()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(NEG_LOW,NEG_HIGH);
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


// Inner loop
void convert_dataset_item(const char* image_filename_left,
                          const char* image_filename_right,
                          const char* image_filename_disp,
                          leveldb::DB* db,
                          int& count)
{
    cv::Mat img_left = cv::imread(image_filename_left,
                                  cv::IMREAD_GRAYSCALE);
    assert(!img_left.empty() && img_left.isContinuous());
    cv::Mat img_right = cv::imread(image_filename_right,
                                   cv::IMREAD_GRAYSCALE);
    assert(!img_right.empty() && img_right.isContinuous());

    cv::Mat img_disp = cv::imread(image_filename_disp,
                                  cv::IMREAD_GRAYSCALE);
    assert(!img_disp.empty() && img_disp.isContinuous());

    int rows = img_left.rows;
    int cols = img_left.cols;


    //random generators
    Random positive_generator(time(NULL), -POS, POS);
    Random negtive_generator(time(NULL), NEG_LOW, NEG_HIGH);
    // Iterate every point within border
    for(int r = 0; r < rows; r+=25)
    {
        uchar* disp_data= img_disp.ptr<uchar>(r);
        for(int c = 0 ; c < cols; c+=25)
        {
            //std::cout << c << '\t' << r << std::endl;
            // Read disparity
            uchar d = disp_data[c];
            if(d==0)
                continue;
            // Create negtive sample
            {
                std::string buffer( 2*PIXEL_COUNT, ' ');
                std::string value;

                caffe::Datum datum;
                datum.set_channels(2);  // one channel for each image in the pair
                datum.set_height(PATCH_SIZE);
                datum.set_width(PATCH_SIZE);
                datum.set_encoded(false);


                cv::Mat patch_left = crop_image_patch(img_left, c , r);
                if(patch_left.empty() && (!patch_left.isContinuous()))
                    continue;
                for (int h = 0; h < PATCH_SIZE; ++h) {
                    const uchar* ptr = patch_left.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < PATCH_SIZE; ++w) {
                        int datum_index = h * PATCH_SIZE + w;
                        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                    }
                }
                // Generate offsets
                int offset = negtive_generator.draw_number();
                cv::Mat patch_right = crop_image_patch(img_right, c - d + offset, r);
                if(patch_right.empty() && (!patch_right.isContinuous()))
                    continue;
                for (int h = 0; h < PATCH_SIZE; ++h) {
                    const uchar* ptr = patch_right.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < PATCH_SIZE; ++w) {
                        int datum_index = PATCH_SIZE + h * PATCH_SIZE + w;
                        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                    }
                }

                datum.set_data(buffer);
                datum.set_label(0);

                datum.SerializeToString(&value);
                assert(value.size() > 0);
                std::string key_str = caffe::format_int(count, 16);
                db->Put(leveldb::WriteOptions(), key_str, value);
                count++;
                //std::cout << count << std::endl;
            }

            // Create positive sample
            {
                std::string value;
                std::string buffer(2*PIXEL_COUNT, ' ');
                caffe::Datum datum;
                datum.set_channels(2);  // one channel for each image in the pair
                datum.set_height(PATCH_SIZE);
                datum.set_width(PATCH_SIZE);
                datum.set_encoded(false);

                cv::Mat patch_left = crop_image_patch(img_left, c , r);
                if(patch_left.empty() && (!patch_left.isContinuous()))
                    continue;
                for (int h = 0; h < PATCH_SIZE; ++h) {
                    const uchar* ptr = patch_left.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < PATCH_SIZE; ++w) {
                        int datum_index =  h * PATCH_SIZE + w;
                        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                    }
                }
                // Generate offsets
                int offset = positive_generator.draw_number();
                cv::Mat patch_right = crop_image_patch(img_right, c - d + offset, r);
                if(patch_right.empty() && (!patch_right.isContinuous()))
                    continue;


                for (int h = 0; h < PATCH_SIZE; ++h) {
                    const uchar* ptr = patch_right.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < PATCH_SIZE; ++w) {
                        int datum_index = PATCH_SIZE + h * PATCH_SIZE + w;
                        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                    }
                }
                datum.set_data(buffer);
                datum.set_label(1);

                datum.SerializeToString(&value);
                assert(value.size() > 0);
                std::string key_str = caffe::format_int(count, 16);
                db->Put(leveldb::WriteOptions(), key_str, value);
                count++;
                //std::cout << count << std::endl;
            }
        }
    }

    return;
}

// Conveision procedure
void convert_dataset(const char* imgs_dir_left,
                     const char* imgs_dir_right,
                     const char* imgs_dir_disp,
                     const char* db_filename_train,
                     const char* db_filename_test)
{
    std::vector<std::string> img_filenames_left;
    std::vector<std::string> img_filenames_right;
    std::vector<std::string> img_filenames_disp;

    int num_imgs = 0;

    // Open leveldb

    //img_filenames_left = list_filename(imgs_dir_left);
    //img_filenames_right = list_filename(imgs_dir_right);
    img_filenames_disp = list_filename(imgs_dir_disp);
    num_imgs = img_filenames_disp.size();
    size_t slpit = 0.6f * num_imgs;
    std::vector<std::string> img_filenames_disp_train;
    std::vector<std::string> img_filenames_disp_test;

    for(int itemid = 0; itemid < slpit; ++itemid)
    {
        img_filenames_disp_train.push_back(img_filenames_disp[itemid]);
    }
    for(int itemid = slpit; itemid < num_imgs; ++itemid)
    {
        img_filenames_disp_test.push_back(img_filenames_disp[itemid]);
    }

    leveldb::DB* db_train;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
                options, db_filename_train, &db_train);
    CHECK(status.ok()) << "Failed to open leveldb " << db_filename_train
                       << ". Is it already existing?";
    for(int itemid = 0; itemid < img_filenames_disp_train.size(); ++itemid)
    {
        fs::path disp_path(img_filenames_disp_train[itemid]);
        std::string img_left = imgs_dir_left;
        img_left.append(fs::basename(disp_path));
        img_left.append(".png");
        img_filenames_left.push_back(img_left);
        std::string img_right = imgs_dir_right;
        img_right.append(fs::basename(disp_path));
        img_right.append(".png");
        img_filenames_right.push_back(img_right);
    }

    int count = 0;
    boost::progress_display progress(img_filenames_disp_train.size());
    for(int itemid = 0; itemid < img_filenames_disp_train.size(); ++itemid)
    {
        convert_dataset_item(img_filenames_left[itemid].c_str(),
                             img_filenames_right[itemid].c_str(),
                             img_filenames_disp_train[itemid].c_str(),
                             db_train,
                             count);
        ++progress;
    }
    delete db_train;

    img_filenames_left.clear();
    img_filenames_right.clear();

    leveldb::DB* db_test;
    status = leveldb::DB::Open(
                options, db_filename_test, &db_test);
    CHECK(status.ok()) << "Failed to open leveldb " << db_filename_test
                       << ". Is it already existing?";

    for(int itemid = 0; itemid < img_filenames_disp_test.size(); ++itemid)
    {
        fs::path disp_path(img_filenames_disp_test[itemid]);
        std::string img_left = imgs_dir_left;
        img_left.append(fs::basename(disp_path));
        img_left.append(".png");
        img_filenames_left.push_back(img_left);

        std::string img_right = imgs_dir_right;
        img_right.append(fs::basename(disp_path));
        img_right.append(".png");
        img_filenames_right.push_back(img_right);
    }

    count = 0;
    progress.restart(img_filenames_disp_test.size());
    for(int itemid = 0; itemid < img_filenames_disp_test.size(); ++itemid)
    {
        convert_dataset_item(img_filenames_left[itemid].c_str(),
                             img_filenames_right[itemid].c_str(),
                             img_filenames_disp_test[itemid].c_str(),
                             db_test,
                             count);
        ++progress;
    }

    delete db_test;
}

int main(int argc,char** argv)
{
    if(argc < 3)
        return 1;
    const char* imgs_dir = argv[1];
    std::string imgs_dir_left = imgs_dir;
    imgs_dir_left.append("/image_2/");

    std::string imgs_dir_right = imgs_dir;
    imgs_dir_right.append("/image_3/");

    std::string imgs_dir_disp = imgs_dir;
    imgs_dir_disp.append("/disp_noc_0/");

    const char* db_filename = argv[2];
    std::string db = db_filename;
    fs::path db_path(db_filename);
    if(fs::exists(db_path))
        fs::remove_all(db_path);
    fs::create_directory(db_path);
    std::string db_train = db + "/train";
    std::string db_test = db + "/test";
    convert_dataset(imgs_dir_left.c_str(),
                    imgs_dir_right.c_str(),
                    imgs_dir_disp.c_str(),
                    db_train.c_str(),
                    db_test.c_str());
    return 0;
}
