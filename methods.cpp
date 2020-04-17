#include "methods.h"
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

// Otsu's
//-------------------------------------------------------------------------------------------------------

// Алгоритм предполагает , что изображение содержит два класса пикселей следующих би-модальных гистограмм 
// (передний план пикселей и фоновые пикселов), он затем вычисляет оптимальный порог , разделяющий два класса

float helper_in_Otsus_method(int hist[], int level, float v, int pixels_n)
// просто считаем коэффициенты ддля взвешенной суммы, по формулам из отчета
{
	long int total = pixels_n * v, m = 0, rem;
	float w0, w1, u0, u1;
	int n = 0, rem_pixels;

	for (int i = 0; i < level; i++)
	{
		m += i * hist[i];
		n += hist[i];
	}

	rem = total - m;
	rem_pixels = pixels_n - n;

	w0 = (1.0 * n) / (1.0 * pixels_n);
	w1 = (1.0 * rem) / (1.0 * pixels_n);
	u0 = (1.0 * m) / (1.0 * n);
	u1 = (1.0 * rem) / (1.0 * rem_pixels);

	// Метод Оцу ищет порог, уменьшающий дисперсию внутри класса, которая определяется 
	// как взвешенная сумма дисперсий двух классов:
	return (w0 * w1 * (u0 - u1) * (u0 - u1));
}

Mat Otsus_method(Mat &input, Mat &output)
{
    long int u = 0;
    int hist[256], size = input.rows * input.cols, n, pixels_n = size, treshold = 0;
    float v, max = 0, x;
    for (int i; i< 256; i++)
        hist[i] = 0;
    cvtColor(input, input, cv::COLOR_RGB2GRAY);

    output = input.clone();

    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            n = output.at<uchar>(i, j);  // значение пикселя
            u += n;
            hist[n]++;  // Вычисляем гистограмму
        }
    }

    v = (1.0 * u) / float(pixels_n);  // среднее значение

    for (int i = 1; i < 255; i++)
    {
        x = helper_in_Otsus_method(hist, i, v, pixels_n);
        if (x > max)  // если полученная взвешенная сумма больше, чем имеющееся, то запоминаем ее
                      // и значение порога treshold
        {
            max = x;
            treshold = i;
        }
    }

    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            if (output.at<uchar>(i, j) > treshold)
                output.at<uchar>(i, j) = 255;
            else output.at<uchar>(i, j) = 0;
        }
    }

    imwrite("otsu.jpg", output);
    return output;
}

//-------------------------------------------------------------------------------------------------------
void k_means_method(Mat &input, Mat &output) 
{
	// in file "k_means.cpp"
}

Mat Inversion(Mat src)
{
    for (int i=1;i<src.rows;i++)
    {
        for (int j=1;j<src.cols;j++)
        {
         src.at<cv::Vec3b>(i,j)[0]=255-src.at<cv::Vec3b>(i,j)[0];
         src.at<cv::Vec3b>(i,j)[1]=255-src.at<cv::Vec3b>(i,j)[1];
         src.at<cv::Vec3b>(i,j)[2]=255-src.at<cv::Vec3b>(i,j)[2];
        }
    }
    return src;
}

Mat Hi_light(Mat src)
{
    for (int i=1;i<src.rows;i++)
    {
        for (int j=1;j<src.cols;j++)
        {
         src.at<cv::Vec3b>(i,j)[0]=10+src.at<cv::Vec3b>(i,j)[0];
         src.at<cv::Vec3b>(i,j)[1]=10+src.at<cv::Vec3b>(i,j)[1];
         src.at<cv::Vec3b>(i,j)[2]=10+src.at<cv::Vec3b>(i,j)[2];
        }
    }
    return src;
}


Mat Sepia(Mat src)
{
    for (int i=1;i<src.rows;i++)
    {
        for (int j=1;j<src.cols;j++)
        {
         float tr = 0.393 * src.at<cv::Vec3b>(i,j)[0] + 0.769 * src.at<cv::Vec3b>(i,j)[1] + 0.189 * src.at<cv::Vec3b>(i,j)[2];
         float tg = 0.349 * src.at<cv::Vec3b>(i,j)[0] + 0.686 * src.at<cv::Vec3b>(i,j)[1] + 0.168 * src.at<cv::Vec3b>(i,j)[2];
         float tb = 0.272 * src.at<cv::Vec3b>(i,j)[0] + 0.534 * src.at<cv::Vec3b>(i,j)[1] + 0.131 * src.at<cv::Vec3b>(i,j)[2];
         if (tr<255)
             src.at<Vec3b>(i,j)[2]=tr;
         else
             src.at<Vec3b>(i,j)[2]=255;
         if (tg<255)
             src.at<Vec3b>(i,j)[1]=tg;
         else
             src.at<Vec3b>(i,j)[1]=255;
         if (tb<255)
             src.at<Vec3b>(i,j)[0]=tb;
         else
             src.at<Vec3b>(i,j)[0]=255;
        }
    }
    return src;
}

Mat Gray_sc(Mat input)
{
    for (int i=1;i<input.rows;i++)
    {
        for (int j=1;j<input.cols;j++)
        {
         int gray_val = 0.299*input.at<cv::Vec3b>(i,j)[0]+0.587*input.at<cv::Vec3b>(i,j)[1]+0.114*input.at<cv::Vec3b>(i,j)[2];
         input.at<cv::Vec3b>(i,j)[0]=gray_val;
         input.at<cv::Vec3b>(i,j)[1]=gray_val;
         input.at<cv::Vec3b>(i,j)[2]=gray_val;
        }
    }
    return input;
}

Mat Gray_world(Mat src)
{
    int R=0, G=0, B=0, N=0;
      for (int i=0;i<src.rows;i++)
      {
          for (int j=0;j<src.cols;j++)
          {
            B += src.at<cv::Vec3b>(i,j)[0];
            G += src.at<cv::Vec3b>(i,j)[1];
            R += src.at<cv::Vec3b>(i,j)[2];
            std::cout<<B<<std::endl;
            N++;
          }
      }
      B= B/N;
       std::cout<<B<<std::endl;
      G= G/N;
           std::cout<<G<<std::endl;
      R= R/N;
           std::cout<<R<<std::endl;
      int gray_val = (R+G+B)/ (3);
      std::cout<<gray_val<<std::endl;

      for (int i=0;i<src.rows;i++)
      {
          for (int j=0;j<src.cols;j++)
          {
              src.at<cv::Vec3b>(i,j)[0]=src.at<cv::Vec3b>(i,j)[0]* gray_val / B;
              src.at<cv::Vec3b>(i,j)[1]=src.at<cv::Vec3b>(i,j)[1]* gray_val / G;
              src.at<cv::Vec3b>(i,j)[2]=src.at<cv::Vec3b>(i,j)[2]* gray_val / R;
          }
      }
    return src;
}
//-------------------------------------------------------------------------------------------------------
Mat Gauss(Mat input)
{
    Mat output(input);

    MyGaussianBlur gB = MyGaussianBlur(input);
    output = gB.process();

    return output;
}

//-------------------------------------------------------------------------------------------------------
Mat Sobel(Mat input)
{
    cvtColor(input, input, cv::COLOR_BGR2GRAY);
    Mat output = input.clone();

    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};

    int Gy[3][3] = {{-1, -2, -1},
                    {0, 0, 0},
                    {1, 2, 1}};
    int Gx_s = 0;
    int Gy_s = 0;
    int G = 0;

    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
          for(int a = 0; a < 3; a++) {
            for(int b = 0; b < 3; b++) {
                Gx_s += Gx[a][b] * input.at<uchar>(i + a - 1, j + b - 1);
                Gy_s += Gy[a][b] * input.at<uchar>(i + a - 1, j + b - 1);
            }
          }
          G = abs(Gx_s) + abs(Gy_s);
          G = (G > 255) ? 255 : G;
          G = (G < 0) ? 0 : G;
          output.at<uchar>(i, j) = G;
          Gx_s = 0;
          Gy_s = 0;
        }
    }
    return output;
}

Mat Blur(Mat input)
{
//    cvtColor(input, input, cv::COLOR_BGR2GRAY);
    Mat output = input.clone();

    float Gx[3][3] = {{0.33, 0.33, 0.33},
                    {0.33, 0.33, 0.33},
                    {0.33, 0.33, 0.33}};

    float Gx_s_g = 0, Gx_s_r = 0, Gx_s_b = 0;
    float G_r = 0, G_b=0, G_g=0;

    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
          for(int a = 0; a < 3; a++) {
            for(int b = 0; b < 3; b++) {
                Gx_s_r += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[2];
                Gx_s_g += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[1];
                Gx_s_b += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[0];
            }
          }
          G_r = abs(Gx_s_r);
          G_b = abs(Gx_s_b);
          G_g = abs(Gx_s_g);
          G_r = (G_r > 255) ? 255 : G_r;
          G_g = (G_g > 255) ? 255 : G_g;
          G_b = (G_b > 255) ? 255 : G_b;

          output.at<Vec3b>(i, j)[0] = G_b;
          output.at<Vec3b>(i, j)[1] = G_g;
          output.at<Vec3b>(i, j)[2] = G_r;
          Gx_s_b=0;
          Gx_s_g=0;
          Gx_s_r=0;

        }
    }
    return output;
}

Mat Median(Mat input)
{
//    cvtColor(input, input, cv::COLOR_BGR2GRAY);
    Mat output = input.clone();

    float Gx[3][3] = {{1,1,1},
                    {1,1,1},
                    {1,1,1}};

    float Gx_s_g = 0, Gx_s_r = 0, Gx_s_b = 0;
    float G_r = 0, G_b=0, G_g=0;

    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
          for(int a = 0; a < 3; a++) {
            for(int b = 0; b < 3; b++) {
                Gx_s_r += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[2];
                Gx_s_g += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[1];
                Gx_s_b += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[0];
            }
          }
          G_r = abs(Gx_s_r);
          G_b = abs(Gx_s_b);
          G_g = abs(Gx_s_g);
          G_r = (G_r > 255) ? 255 : G_r;
          G_g = (G_g > 255) ? 255 : G_g;
          G_b = (G_b > 255) ? 255 : G_b;

          output.at<Vec3b>(i, j)[0] = G_b;
          output.at<Vec3b>(i, j)[1] = G_g;
          output.at<Vec3b>(i, j)[2] = G_r;
          Gx_s_b=0;
          Gx_s_g=0;
          Gx_s_r=0;

        }
    }
    return output;
}


Mat Contrast(Mat input)
{
//    cvtColor(input, input, cv::COLOR_BGR2GRAY);
    Mat output = input.clone();

    float Gx[3][3] = {{-1, -1, -1},
                    {-1, 9, -1},
                    {-1, -1, -1}};

    float Gx_s_g = 0, Gx_s_r = 0, Gx_s_b = 0;
    float G_r = 0, G_b=0, G_g=0;

    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
          for(int a = 0; a < 3; a++) {
            for(int b = 0; b < 3; b++) {
                Gx_s_r += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[2];
                Gx_s_g += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[1];
                Gx_s_b += Gx[a][b] * input.at<Vec3b>(i + a - 1, j + b - 1)[0];
            }
          }
          G_r = abs(Gx_s_r);
          G_b = abs(Gx_s_b);
          G_g = abs(Gx_s_g);
          G_r = (G_r > 255) ? 255 : G_r;
          G_g = (G_g > 255) ? 255 : G_g;
          G_b = (G_b > 255) ? 255 : G_b;

          output.at<Vec3b>(i, j)[0] = G_b;
          output.at<Vec3b>(i, j)[1] = G_g;
          output.at<Vec3b>(i, j)[2] = G_r;
          Gx_s_b=0;
          Gx_s_g=0;
          Gx_s_r=0;

        }
    }
    return output;
}

Mat Tes(Mat input)
{
    cvtColor(input, input, cv::COLOR_BGR2GRAY);
    Mat output = input.clone();

    float Gx[3][3] = {{0, 1, 0},
                    {1, 0, -1},
                    {0, -1, 0}};

    float Gx_s_g = 0, Gx_s_r = 0, Gx_s_b = 0;
    float G_r = 0, G_b=0, G_g=0;

    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
          for(int a = 0; a < 3; a++) {
            for(int b = 0; b < 3; b++) {
                Gx_s_r += Gx[a][b] * input.at<uchar>(i + a - 1, j + b - 1);
            }
          }
          G_r = (Gx_s_r+255)/2;
          G_r = (G_r > 255) ? 255 : G_r;

          output.at<uchar>(i, j) = G_r;
          Gx_s_r=0;

        }
    }
    return output;
}


//-------------------------------------------------------------------------------------------------------
int minimum(std::vector<int> *grayscale_image) {
  return *std::min_element(grayscale_image->begin(), grayscale_image->end());
}

int maximum(std::vector<int> *grayscale_image) {
  return *std::max_element(grayscale_image->begin(), grayscale_image->end());
}


int use_formula(int value, int max_value, int min_value) {
  if (max_value < min_value)
    throw -1;

  float a = (-1) * (static_cast<float>(255) / (max_value - min_value)) * min_value;
  float b = static_cast<float>(255) / (max_value - min_value);

  if (max_value == min_value)
    return 0;
  else
    return (a + b * value);
}

std::vector <int> contrast_increase(std::vector<int> grayscale_image, int width, int height) {
  int size = width * height, error = 0;

  if (size <= 0) {
    error = 1;
  }
  switch (error) {
    case 1:
      throw std::runtime_error("size <= 0");
  }
  std::vector<int> output(grayscale_image);

  int min = minimum(&grayscale_image);
  int max = maximum(&grayscale_image);

  for (int i = 0; i < size; i++) {
    output[i] = use_formula(grayscale_image[i], max, min);
  }
  return output;
}

Mat histStretch(Mat src)
{
    int gray[256] {0};
    Mat dst(src);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            gray[(int)src.at<uchar>(y, x)]++;
        }
    }
    std::vector<int> fake = std::vector<int>(src.rows * src.cols);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            fake[y * src.cols + x] = (int)src.at<uchar>(y, x);
        }
    }
    std::vector<int> out = contrast_increase(fake, src.cols, src.rows);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            dst.at<uchar>(y, x) = (uchar)out[y * src.cols + x];
        }
    }
    return dst;
}

Mat Linear_contrast(Mat input)
{
    Mat out(input);
    out = histStretch(input);

    return out;
}
//-------------------------------------------------------------------------------------------------------
