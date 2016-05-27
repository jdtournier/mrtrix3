/*
 * Copyright (c) 2008-2016 the MRtrix3 contributors
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/
 * 
 * MRtrix is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * For more details, see www.mrtrix.org
 * 
 */


#include "command.h"
#include "image.h"
#include <Eigen/Dense>

#define DEFAULT_SIZE 5

using namespace MR;
using namespace App;


void usage ()
{
  DESCRIPTION
  + "denoise DWI data and estimate the noise level based on the optimal threshold for PCA.";

  
  AUTHOR = "Daan Christiaens (daan.christiaens@kuleuven.be) & Jelle Veraart (jelle.veraart@nyumc.org) & J-Donald Tournier (jdtournier.gmail.com)";
  
  
  ARGUMENTS
  + Argument ("dwi", "the input diffusion-weighted image.").type_image_in ()

  + Argument ("out", "the output denoised DWI image.").type_image_out ();


  OPTIONS
  + Option ("size", "set the window size of the denoising filter. (default = " + str(DEFAULT_SIZE) + ")")
    + Argument ("window").type_integer (0, 50)
  
  + Option ("noise", "the output noise map.")
    + Argument ("level").type_image_out();

}


typedef float value_type;


template <class ImageType>
class DenoisingFunctor
{
public:
  DenoisingFunctor (ImageType& dwi, int size)
    : extent(size/2),
      m(dwi.size(3)),
      n(size*size*size),
      X(m,n)
  { }
  
  void operator () (ImageType& dwi, ImageType& out)
  {
    load_data(dwi);
    std::cout << X << std::endl;
  }
  
  void load_data (ImageType& dwi)
  {
    pos = { dwi.index(0), dwi.index(1), dwi.index(2) };
    X.setZero();
    ssize_t k = 0;
    for (dwi.index(2) -= extent; dwi.index(2) <= pos[2]+extent; ++dwi.index(2), ++k)
      for (dwi.index(1) -= extent; dwi.index(1) <= pos[1]+extent; ++dwi.index(1), ++k)
        for (dwi.index(0) -= extent; dwi.index(0) <= pos[0]+extent; ++dwi.index(0), ++k)
          if (dwi.valid())
            X.column(k) = dwi.row(3);
    // reset image position
    dwi.index(0) = pos[0];
    dwi.index(1) = pos[1];
    dwi.index(2) = pos[2];
  }
  
private:
  int extent;
  size_t m, n;
  Eigen::MatrixXd X;
  ssize_t pos[] = {0, 0, 0};
  
};



void run ()
{
  auto dwi_in = Image<value_type>::open (argument[0]).with_direct_io(3);

  auto header = Header (dwi_in);
  header.datatype() = DataType::Float32;
  auto dwi_out = Image<value_type>::create (argument[1], header);
  
  int extent = get_option_value("size", DEFAULT_SIZE);
  
  DenoisingFunctor< Image<value_type> > func (dwi_in, extent);
  
  dwi_in.index(0) = 10;
  dwi_in.index(1) = 10;
  dwi_in.index(2) = 10;
  
  func(dwi_in, dwi_out);
  

}


