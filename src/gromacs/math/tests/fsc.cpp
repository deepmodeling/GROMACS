#include "gmxpre.h"

#include "gromacs/math/fsc.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/mdspan/extensions.h"

namespace gmx
{

namespace test
{

namespace
{

template<typename U, typename T>
void print3DData(basic_mdspan<const T, dynamicExtents3D> input, U printFunction)
{
    fprintf(stderr,"\n");
    fprintf(stderr,"-------\n");
    for (ptrdiff_t x = 0; x< input.extent(XX); ++x)
    {
        for (ptrdiff_t y = 0; y< input.extent(YY); ++y)
        {
            for (ptrdiff_t z = 0; z< input.extent(ZZ); ++z)
            {
                printFunction(input(x,y,z));
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"-------\n");
    }
    fprintf(stderr,"\n");

}

TEST(FSC, Gaussian)
{
    MultiDimArray<std::vector<float>, dynamicExtents3D> x(3,3,3);
    
    const float        center         = 0.0634936392307281494140625;      // center
    const float        f              = 0.0385108403861522674560546875;   // face
    const float        e              = 0.0233580060303211212158203125;   // edge
    const float        c              = 0.014167346991598606109619140625; // corner
    std::vector<float> data = { c * e, e * e , c * e , e, f, e, c * e , e, c,
                                e, f, e, f, center, f, e, f, e,
                                c, e, c, e, f,      e, c, e, c };
    std::copy(std::begin(data), std::end(data), begin(x));
    
    const auto printRealFunction = [](const float & value){
            fprintf(stderr,"(%10.4g) \t", value);};
    print3DData(x.asConstView(), printRealFunction);

    FourierTransform fft({3,3,3});

    auto result = fft.transform(x);

    const auto printFunction = [](const t_complex & x){fprintf(stderr,"(%10.4g %10.4g) \t", x.re, x.im);};
    print3DData(result, printFunction);
}

TEST(FSC, smallCube)
{
    MultiDimArray<std::vector<float>, dynamicExtents3D> x(2,2,2);
    
    const float        center         = 0.0634936392307281494140625;      // center
    const float        f              = 0.0385108403861522674560546875;   // face
    const float        e              = 0.0233580060303211212158203125;   // edge
    const float        c              = 0.014167346991598606109619140625; // corner
    std::vector<float> data = { e,f,c,center,e * e,f *f ,c *c,center * center };
    std::copy(std::begin(data), std::end(data), begin(x));
    
    const auto printRealFunction = [](const float & value){
            fprintf(stderr,"(%10.4g) \t", value);};
    print3DData(x.asConstView(), printRealFunction);

    FourierTransform fft({2,2,2});

    auto result = fft.transform(x);

    const auto printFunction = [](const t_complex & x){fprintf(stderr,"(%10.4g %10.4g) \t", x.re, x.im);};
    print3DData(result, printFunction);
}


TEST(FSC, sizes)
{
    FourierTransform fft({3,3,3});
    fprintf(stderr,"\n\n %d %d %d\n\n",fft.realSize()[XX], fft.realSize()[YY], fft.realSize()[ZZ]);
    fprintf(stderr,"\n\n %d %d %d\n\n",fft.complexSize()[XX], fft.complexSize()[YY], fft.complexSize()[ZZ]);

    FourierTransform fft2({3,5,7});
    fprintf(stderr,"\n\n %d %d %d\n\n",fft2.realSize()[XX], fft2.realSize()[YY], fft2.realSize()[ZZ]);
    fprintf(stderr,"\n\n %d %d %d\n\n",fft2.complexSize()[XX], fft2.complexSize()[YY], fft2.complexSize()[ZZ]);


    FourierTransform fft3({7,5,3});
    fprintf(stderr,"\n\n %d %d %d\n\n",fft3.realSize()[XX], fft3.realSize()[YY], fft3.realSize()[ZZ]);
    fprintf(stderr,"\n\n %d %d %d\n\n",fft3.complexSize()[XX], fft3.complexSize()[YY], fft3.complexSize()[ZZ]);

    FourierTransform fft4({8,5,3});
    fprintf(stderr,"\n\n %d %d %d\n\n",fft4.realSize()[XX], fft4.realSize()[YY], fft4.realSize()[ZZ]);
    fprintf(stderr,"\n\n %d %d %d\n\n",fft4.complexSize()[XX], fft4.complexSize()[YY], fft4.complexSize()[ZZ]);

}

}
}
}