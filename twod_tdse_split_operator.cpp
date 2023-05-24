
/*
 * two dimensional time dependent schrodinger equation using split operator
 * icc -Wall -O2 -std=c++17 -O3 -DNDEBUG -funroll-loops -ffast-math -lstdc++ -fopenmp fftwpp/fftw++.cc -lfftw3 -lfftw3_omp -lm twod_tdse_split_operator.cpp -lm -no-multibyte-chars
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <iomanip> //this is for setprecision
#include "fftwpp/Array.h"
#include "fftwpp/fftw++.h"

using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace std;

typedef std::complex<double> dcompx;
typedef std::vector<double> dvec;
typedef std::vector<dcompx> dcopxvec;
typedef std::vector<std::vector<dcompx> > ddcopxvec;
constexpr dcompx I{0.0, 1.0};
#define pi 3.1415926535897932384626433
#define imax(a,b) ((a>b)?a:b)




class TwoD_TDSE{

    public:
            TwoD_TDSE();
            ~TwoD_TDSE();
            
            void MaskFunction(array2<Complex> &psi2d, double prct);
            void getExceptation_value(array2<Complex> &psi2d,array2<Complex> &psi2d_fft, double t, double E0);
            Complex getCoefficient(array2<Complex> &psi2d_0, array2<Complex> &psi2d);
            double integral_grandstate_energy(array2<Complex> &psi2d, int iR);
            void Normalizer(array2<Complex> &psi2d);
            double potential(double x, double y, int iR);
            template <class T>
            std::string cstr(T value);
            double E_x(int t_in, int iA);
            double E_y(int t_in, int iA);
            void twod_tidse();
            
            
            //prameters
            
            //spatial space
            double dx = 0.2;
            double x_min = -100.0;
            double x_max = -x_min;
            unsigned int Nx = ((x_max - x_min)/dx);

            double dy = 0.2;
            double y_min = -100.0;
            double y_max = -y_min;
            unsigned int Ny = ((y_max - y_min)/dy);
            
            //momentum space
            double dk_x = 2.0*pi/(Nx*dx);
            double k0_x = -0.5*Nx*dk_x;
            
            double dk_y = 2.0*pi/(Ny*dy);
            double k0_y = -0.5*Ny*dk_y;
            
            //time space
            double t_min = -300.0;
            double t_max = 300.0;
            double dt = 0.15;//(t_max - t_min)/N_time;
            int N_time = int((t_max - t_min)/dt)+1;
            
            int iA = 0.0;

            double angle_x = 0.0;
            double angle_y = pi/2.0;
            
            double e = 0.0;
            
            //the number of OpenMP threads
            int NTHREADS = omp_get_max_threads();

};

TwoD_TDSE::TwoD_TDSE(){
}

TwoD_TDSE::~TwoD_TDSE(){
}

double TwoD_TDSE::potential(double x, double y, int iR){

    double eps=0.6384;

    //return -0.5*(x*x+y*y);//-1.0/sqrt(x*x+2.0);
    //return -2.0/sqrt(y*y+0.5)-2.0/sqrt(x*x+0.5)+1.0/sqrt((y-x)*(y-x)+0.339);
    
    //return -1.0/sqrt(x*x+y*y+0.6384);
    double R=1.0;
    double xc = R/2.0;
    double yc = 0.0;
    
    
    double alp = 2.1325;
    double v4 = -(1.00+exp(-alp*((y)*(y)+(x)*(x))))/sqrt((y)*(y)+(x)*(x)+0.005);
    
    //return -1.0/sqrt((x-xc)*(x-xc)+(y)*(y)+eps)-1.0/sqrt((x+xc)*(x+xc)+(y)*(y)+eps)-1.0/sqrt((y+2*xc)*(y+2*xc)+(x)*(x)+eps);
    
    //return -exp(-0.3*((y)*(y)+(x)*(x)))/sqrt((y)*(y)+(x)*(x)+0.005);
    //return v4;
    
    //return -1.0/sqrt((x-xc)*(x-xc)+(y)*(y)+4.33)-1.0/sqrt((x+xc)*(x+xc)+(y)*(y)+4.33); //h2+

    double v20 =  -exp( -0.3*( (x-R/2.0)*(x-R/2.0) + (y)*(y) ) )\
                    -exp( -0.3*( (x+2.0*R)*(x+2.0*R) + (y)*(y) ) )\
                          -exp( -0.3*( (x)*(x) + (y)*(y) ) );
                          
    double b = -0;
    double a = -1.5;
    double c = -1.5;
    double v21 = -exp( -0.3*( (x-R/2.0)*(x-R/2.0) + (y)*(y) ) ) * ( (y/a)*(y/a) + (y/b)*(y/b) );
    double v22 = -0.7*exp(-0.1*(2*(x*x)+0.8*(y*y)+0.2*x*y))-exp(-0.9*((x-a)*(x-a)+(y-b)*(y-b)));//-0.5*exp(-0.9*((x+c)*(x+c)+(y+b)*(y+b)));
    
    double vv0 = -1.0;
    double theta = 1.6;// M_PI/8.0;
    double r = sqrt(x*x+y*y);
    double phi = atan2(y,x);
    double v23 = vv0*(x*x+y*y - 2*x*y + 3.0*( r*cos(phi)*sin(theta) + cos(theta) )*(r*cos(phi)*sin(theta) + cos(theta) ) )/pow(r*r+1.0,2.5);


    double F = 1.0, E = 0.2, D = 2., C = 0.1, B = 0., A = 1.8;
    double v24 = -exp(-F*(E*(x-D)*(x-D)+C*(y-B)*(y-B)+A*x*y));



    double a0 = 1.;
    double a1 = 1.;
    double a2 = 0.;
    double a3 = -2.0+R;

    double epsi = 0.5;

    double v25 = -1.0*exp(-(a0*x*x+a1*y*y+a2*x+a3*x*y))/pow( (r+epsi),2.5 );

        
    double RR = 30, aa = 0.05;
    double v27 = - 2.0/(1.0 + exp((r - RR)/aa));
    double v28 = - 2.0*exp(-(r-20)*(r-20));

    //return v27+v28;
        
    double v17 = -exp( -0.3*( (x-R/2.0)*(x-R/2.0) + (y)*(y) ) )\
                    -0.5*exp( -0.3*( (x+R/2.0)*(x+R/2.0) + (y)*(y) ) );
    double v29 = (4.0*x*x-2.0+4.0*y*y-2.0)*exp( -( x*x+y*y-0.5*x )/1.0 );
    
    
    
    double vartheta = M_PI;//2.0;// 10*M_PI/11.0;
    double VV0 = 1.0;
    
    double v13 = VV0*(r*r+1-3.0*pow((r*cos(phi)*sin(vartheta) + cos(vartheta)),2))/pow(r*r+1,2.5);
    
    
    double v30 = x*exp(-0.3*(x*x+y*y));
    
    double v31 = 0.5*x*exp( -0.3*( (x-R/2.0)*(x-R/2.0) + (y)*(y) ) ) + x*exp( -0.3*( (x+R/2.0)*(x+R/2.0) + (y)*(y) ) );
    
    double v32 = -exp( -( (x-R/2.0)*(x-R/2.0) + (y)*(y) )/3. )\
                    -exp( -( (x+R/2.0)*(x+R/2.0) + (y)*(y) )/3. );
    
    double x00 = 1.0;
    
    double v35 = -exp( -( (x-x00)*(x-x00) + (y)*(y) )/3 ) -exp( -( (x-x00)*(x-x00) + (y-x00)*(y-x00) )/3 ) ;
    
    double v36 = -exp(-(x*x+y*y)/3.0);
    
    double v37 =-exp( -( (x-x00)*(x-x00) + (y)*(y) )/3. ) -exp( -( (x-x00)*(x-x00) + (y-x00)*(y-x00) )/3. )\
        -exp(-r*r/3)-exp( -( (x)*(x) + (y+2*x00)*(y+2*x00) )/3. )-exp( -( (x+2*x00)*(x+2*x00) + (y-x00)*(y-x00) )/3. );
  
     double v38 = -exp( -( (x-R/2.0)*(x-R/2.0) + (y)*(y) )/3. )\
                    -exp( -( (x+R/2.0)*(x+R/2.0) + (y)*(y) )/3. );
    return v38;
}

void TwoD_TDSE::twod_tidse(){

    int iR = 20;
    
    std::ifstream inpulse("A_ini.dat");
    inpulse >> iR;
    
    cout << "iR: " << iR <<", OMP nTHREADS: "<<NTHREADS << endl;
    
    fftw::maxthreads=get_max_threads();

    std::ofstream pot_out("potantial.dat");
    for(unsigned int i = 0; i < Nx; i+=1){
        for(unsigned int j = 0; j < Ny; j+=1){
            if(abs(x_min+i*dx) < 100 and abs(y_min+j*dy) < 100)
            pot_out << x_min+i*dx <<"\t"<<y_min+j*dy<<"\t"<<potential(x_min+i*dx,y_min+j*dy,iR)<<endl;
        }
    }pot_out.close();

    //exit(0);    
    size_t align=sizeof(Complex);
    
    array2<Complex> psi2d(Nx,Ny,align);
    
    fft2d Forward(-1,psi2d);
    fft2d Backward(1,psi2d);
    
    cout << "done \n";
    
    //set CHUNK
    int local_NTHREADS = int(NTHREADS);
    int CHUNK = int((Nx)/local_NTHREADS);
    std::ifstream psi_in("../ground_psi2d_"+cstr(int(Nx))+".dat");
    if(!psi_in.is_open()){
        
        cout << "creating the file ..\n";
        for(unsigned int i=0; i < Nx; ++i){
            double x = x_min + i*dx;
            for(unsigned int j=0; j < Ny; j++){
                double y = y_min + j*dy;
                psi2d[i][j]=imax(0.0,1.0-(x*x-1.0)*(x*x-1.0))+imax(0.0,1.0-(y*y-1.0)*(y*y-1.0))*Complex(1.0,0.0);
            }
        }
        
        double dtau = 0.2;
        ofstream eout("ground_energy_"+cstr(int(Nx))+".dat");
        for(int im_t=0; im_t < 5000; ++im_t){

            unsigned int i,j;
            omp_set_num_threads(local_NTHREADS);
            #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j) 
            for(i = 0; i < Nx; ++i){
                for(j = 0; j < Ny; ++j){
                    psi2d[i][j]=exp(-0.5*dtau*potential(x_min+i*dx,y_min+j*dy,iR))*psi2d[i][j];
                }
            }
            Forward.fft(psi2d);
            omp_set_num_threads(local_NTHREADS);
            #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j) 
            for(i = 0; i < Nx; ++i){
                for(j = 0; j < Ny; ++j){
                    psi2d[i][j]=exp( -dtau*( 0.5*(k0_x + i*dk_x)*(k0_x + i*dk_x) + 0.5*(k0_y + j*dk_y)*(k0_y + j*dk_y) ) )*psi2d[i][j];
                }
            }
            
            Backward.fftNormalized(psi2d);
            omp_set_num_threads(local_NTHREADS);
            #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j) 
            for(i = 0; i < Nx; ++i){
                for(j = 0; j < Ny; ++j){
                    psi2d[i][j]=exp(-0.5*dtau*potential(x_min+i*dx,y_min+j*dy,iR))*psi2d[i][j];
                }
            }
            Normalizer(psi2d);
            double instan_energy = integral_grandstate_energy(psi2d, iR);
            
            eout<<im_t<<"\t"<<instan_energy<<"\t"<<(instan_energy)*27.2113961<<"\t"<<endl;
            if(im_t%100==0)cout<<"Img_time  = "<<im_t<<"\t ground_energy = "<<instan_energy<<"au,\t"<<(instan_energy)*27.2113961<<"eV"<<endl;

        }
        
        ofstream fout("ground_psi2d_"+cstr(int(Nx))+".dat");
        for(unsigned int i = 0; i < Nx; ++i){
            for(unsigned int j = 0; j < Ny; ++j){
                fout<<x_min+i*dx <<"\t"<<y_min+j*dy<<"\t"<<real(psi2d[i][j])<<endl;
            }
        }
    
    }
    else{
        cout << "reading the file ..\n";
        ddcopxvec tE;

        double x_in,y_in,abs_psi,real_psi,imag_psi;
        int il = -1;
        while(psi_in>>x_in>>y_in>>real_psi){
            if(y_in == x_min)il++;
            tE.push_back(dcopxvec());
            tE[il].push_back(real_psi);
        }
        tE.resize(Nx);
        for(int l = 0; l < Nx; l++)
                tE[l].resize(tE[Nx-1].size());

        int len_E = tE[0].size();
        cout <<"ground wave function size: "<< Nx*len_E << "\t" << len_E << endl;
        
        for(unsigned int i = 0; i < Nx; ++i){
            for(unsigned int j = 0; j < Ny; ++j){
                psi2d[i][j] = tE[i][j];
            }
        }
        tE.clear();
        tE.resize(0);
        tE.shrink_to_fit();
    }
    
    double E0 = integral_grandstate_energy(psi2d,iR);
    
    double local_w0 = 0.3 - E0;
    iA = int(local_w0/0.01);
    this->iA = iA;
    
    ofstream plot_psiG("plot_ground_psi2d.dat");
    for(unsigned int i = 0; i < Nx; i++){
        for(unsigned int j = 0; j < Ny; j++){
            if(abs(x_min+i*dx) < 100 and abs(y_min+j*dy) < 100)
                plot_psiG<<x_min+i*dx <<"\t"<<y_min+j*dy<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
        }
    }plot_psiG.close();
    
    
    
    cout << "ground E0: " << E0 << " iA:"<<iA << " w0:"<< local_w0 << endl;
    array2<Complex> psi2d_0(Nx,Ny,align);
    for(unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            psi2d_0[i][j] = psi2d[i][j];
        }
    }
    
    
    dvec pulsex(N_time,0.0);
    dvec pulsey(N_time,0.0);
    for(int i = 0; i < N_time; i++){
        pulsex[i] = E_x(i,iA);
        pulsey[i] = E_y(i,iA);
     }
    ofstream potout("pulse_x.dat");
    for(int i = 0; i < N_time; i++){
        potout << t_min+i*dt<<"\t"<<pulsex[i]<<"\t"<<pulsey[i]<<endl;
    }
    
    
    //exit(0);
    cout<<"Foreward time propagation ...\n";cout << "N time: "<<N_time << "\t dt: "<<dt << endl;
    for(int t=0; t<N_time; ++t){

        
        unsigned int i,j;
        omp_set_num_threads(local_NTHREADS);
        #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j)
        for(i = 0; i < Nx; ++i){
            double x_i = x_min+i*dx;
            for(j = 0; j < Ny; ++j){
                double y_j = y_min+j*dy;
                dcompx pot_exp_part = ( cos(-0.5*dt*(potential(x_i,y_j,iR)+( x_i*pulsex[t]+y_j*pulsey[t] ) ) ) + I* sin(-0.5*dt*(potential(x_i,y_j,iR)+( x_i*pulsex[t]+y_j*pulsey[t] ) )  )  );
                psi2d[i][j]=pot_exp_part*psi2d[i][j];
            }
        }
        Forward.fft(psi2d);
        omp_set_num_threads(local_NTHREADS);
        #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j)
        for(i = 0; i < Nx; ++i){
            double kx_i = k0_x + i*dk_x;
            for(j = 0; j < Ny; ++j){
                double ky_j = k0_y + j*dk_y;
                dcompx kenetic_exp = (cos(-dt* (0.5*(kx_i)*(kx_i) + 0.5*(ky_j)*(ky_j)) ) + I*sin( -dt* (0.5*(kx_i)*(kx_i) + 0.5*(ky_j)*(ky_j))  )  );
                psi2d[i][j]=kenetic_exp*psi2d[i][j];
            }
        }
        Backward.fftNormalized(psi2d);
        omp_set_num_threads(local_NTHREADS);
        #pragma omp parallel for default(shared) schedule(static, CHUNK) private(j)
        for(i = 0; i < Nx; ++i){
            double x_i = x_min+i*dx;
            for(j = 0; j < Ny; ++j){
                double y_j = y_min+j*dy;
                dcompx pot_exp_part = ( cos(-0.5*dt*(potential(x_i,y_j,iR)+( x_i*pulsex[t]+y_j*pulsey[t]  ) ) ) + I* sin(-0.5*dt*(potential(x_i,y_j,iR)+( x_i*pulsex[t]+y_j*pulsey[t]  ) )  )  );
                psi2d[i][j]=pot_exp_part*psi2d[i][j];
            }
        }
        //MaskFunction(psi2d,26.75);
        if(t%1000==0){
            double sumunit=0.0;for(unsigned int i=0;i<Nx;i++)for(unsigned int j=0;j<Ny;j++)sumunit+=abs(psi2d[i][j])*abs(psi2d[i][j])*dx*dy;
            cout<<"time step = "<<t<<", norm(1)= "<<sumunit<<endl;
        }

    }
    cout << "t4= " << t_min+(N_time-1)*dt << endl;
    ofstream endout("final_psi2d_tf.dat");
    for (unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            endout<<x_min + i*dx<<"\t"<<y_min + j*dy<<"\t"<<real(psi2d[i][j])<<"\t"<<imag(psi2d[i][j])<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
        }
    }endout.close();
    
    
    Complex a_0 = getCoefficient(psi2d_0,psi2d);
    cout << "a0:" << a_0 << "\t |a0|^2:"<<abs(a_0)*abs(a_0)<<endl;
    array2<Complex> psi2d_continumm(Nx,Ny,align);
    array2<Complex> psi2d_fft(Nx,Ny,align);
    for(unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            psi2d_continumm[i][j] = psi2d[i][j]-a_0*psi2d_0[i][j];
            psi2d_fft[i][j] = psi2d[i][j]-a_0*psi2d_0[i][j];
        }
    }
    
    //free(psi2d);
    //free(psi2d_0);
    Forward.fft(psi2d_fft);
    //getExceptation_value(psi2d_continumm,psi2d_fft,t_min+(N_time-1)*dt,E0);
    //free(psi2d_fft);
    Normalizer(psi2d_continumm);
    ofstream endout_con("final_psi2d_continumm.dat");
    for(unsigned int i = 0; i < Nx; i+=20){
        for(unsigned int j = 0; j < Ny; j+=20){
            if( (x_min+i*dx) != 0 and (y_min+j*dy) != 0)
                endout_con<<x_min+i*dx<<"\t"<<y_min+j*dy<<"\t"<<abs(psi2d_continumm[i][j])*abs(psi2d_continumm[i][j])<<"\t"<<abs(psi2d_fft[i][j])*abs(psi2d_fft[i][j])<<endl;
        }
    }endout_con.close();
    //free(psi2d_continumm);
    getExceptation_value(psi2d_continumm,psi2d_fft,t_min+(N_time-1)*dt,E0);
}

//pulse
double TwoD_TDSE::E_x(int t_in, int iA){
    
    double t = t_min+t_in*dt;
    
    
    double k = 0.1;
    double F_0 = sqrt(k/3.51); // the base is 10^16

    double wx = iA*0.01;//52.4/27.211396
    
    double Tx = 2.0*41.341;

    double Ex_t = F_0*exp(-(2.0*log(2.0))*t*t/(Tx*Tx))*cos(t*wx)*1.0/sqrt(e*e+1.0);
    return Ex_t;


}


double TwoD_TDSE::E_y(int t_in, int iA){
    
    double t = t_min+t_in*dt;
    
    
    double k = 0.1;
    double F_0 = sqrt(k/3.51); // the base is 10^16

    double wy = iA*0.01;//52.4/27.211396
    
    double Ty = 2.0*41.341;

    double Ey_t = F_0*exp(-(2.0*log(2.0))*t*t/(Ty*Ty))*sin(t*wy)*e/sqrt(e*e+1.0);
    return Ey_t;

    

}


void TwoD_TDSE::getExceptation_value(array2<Complex> &psi2d,array2<Complex> &psi2d_fft, double t, double E0){
    
    ofstream radial_psi;

    int iip = 0;
    for(double theta = -pi; theta <= pi; theta +=0.005){
        
        radial_psi.open("psiR/radial_psi_"+cstr(iip++)+".dat");
       
        for(unsigned int i=0; i < Nx; ++i){
            double x1 = x_min + i*dx;
            double kx = k0_x + i*dk_x;
            for(unsigned int j=0; j < Ny; j++){
                double y1 = y_min + j*dy;
                double ky = k0_y + j*dk_y;
                double theta1 = atan2(y1,x1);
                //zone 1
                if(theta >= 0. and theta < pi/2.){
                    if(x1 > 0. and y1 > 0.){
                        if(abs( theta1 -theta ) < 0.001){
                            radial_psi <<sqrt(x1*x1+y1*y1)<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
                        }
                        //cout << "zone1 \n";
                    }
                }
                else if(theta >= pi/2.0 and theta <= pi){
                    //zone 2
                    if(x1 < 0. and y1 > 0.){
                        if(abs( theta1 - theta ) < 0.001){
                            radial_psi <<sqrt(x1*x1+y1*y1)<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
                        }
                        //cout  <<theta1<< "  zone2 \n" ;
                    }
                }
                else if(theta >= -pi and theta < -pi/2.0){
                    //zone 3
                    if(x1 < 0. and y1 < 0.){
                        if(abs( theta1 - theta ) < 0.001){
                            radial_psi <<sqrt(x1*x1+y1*y1)<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
                        }
                        //cout << "zone3 \n";
                    }
                }
                else if(theta >= -pi/2. and theta < 0.){
                    //zone 4
                    if(x1 > 0. and y1 < 0.){
                        if(abs( theta1 - theta ) < 0.001){
                            radial_psi <<sqrt(x1*x1+y1*y1)<<"\t"<<abs(psi2d[i][j])*abs(psi2d[i][j])<<endl;
                        }
                        //cout << "zone4 \n";
                    }
                }
            }           
        }radial_psi.close();

    }
   
    
}


Complex TwoD_TDSE::getCoefficient(array2<Complex> &psi2d_0, array2<Complex> &psi2d){
    
    Complex sum = Complex(0.0,0.0);
    for (unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            sum += real(psi2d_0[i][j])*psi2d[i][j];
        }
    }
    
    return sum*dx*dy;
    
}




void TwoD_TDSE::MaskFunction(array2<Complex> &psi2d, double prct){

	double h_abs = abs(x_min)*prct/100.0;
	/*for(int i = 0; i < Nx; ++i){
		double x = x_min + i*dx;
		double fx = pow(cos((abs(x)-abs(x_min) + h_abs)*pi/(2.0*h_abs)),1.0/8.0);
		for(int j = 0; j < Ny; ++j){
			double y = y_min + j*dy;
			double fy = pow(cos((abs(y)-abs(y_min) + h_abs)*pi/(2.0*h_abs)),1.0/8.0);
			if(x > x_max || x < x_min){
				psi[i*Ny+j] = psi[i*Ny+j]*fx;
			}
                        else if(y > y_max || y < y_min){
                                psi[i*Ny+j] = psi[i*Ny+j]*fy;                                                         
                        }
			else{
				psi[i*Ny+j] = psi[i*Ny+j];
			}
		}
	}*/

        for(int i = 0; i < Nx; ++i){
            double x = x_min + i*dx;
            for(int j = 0; j < Ny; ++j){
                    double y = y_min + j*dy;
                    double fxy = tanh(h_abs*(abs(x_min)-sqrt(x*x + y*y)))*tanh(h_abs*(abs(x_min)-sqrt(x*x + y*y)));
                    psi2d[i][j] = psi2d[i][j]*fxy;
            }
        }

}


void TwoD_TDSE::Normalizer(array2<Complex> &psi2d){
    
    double sum = 0.;
    for(unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            sum += abs(psi2d[i][j])*abs(psi2d[i][j]);
        }
    }
    sum=sqrt(sum*dx*dy);
    for(int i = 0; i < Nx; ++i){
        for(int j = 0; j < Ny; ++j){
            psi2d[i][j]=psi2d[i][j]/sum;
        }
    }
}


double TwoD_TDSE::integral_grandstate_energy(array2<Complex> &psi2d,int iR){

    dcompx *ddxpsi = new dcompx[(Nx-1)*(Ny-1)];
    
    for(unsigned int i = 1; i < Nx-1; ++i){
        for(unsigned int j = 1; j < Ny-1; ++j){
            ddxpsi[(i-1)*Ny+(j-1)] = (abs(psi2d[(i+1)][j])-2.0*abs(psi2d[(i)][j])+abs(psi2d[(i-1)][j]))/(dx*dx)\
            + (abs(psi2d[i][(j+1)])-2.0*abs(psi2d[(i)][j])+abs(psi2d[i][(j-1)]))/(dy*dy);
        }
    }
    
    dcompx pot_exp = 0.0;
    for(unsigned int i = 0; i < Nx; ++i){
        for(unsigned int j = 0; j < Ny; ++j){
            pot_exp += psi2d[i][j]*potential(x_min+i*dx,y_min+j*dy,iR)*psi2d[i][j];
        }
    }
    pot_exp = pot_exp*dx*dy;

    dcompx ke_exp = 0.0;
    for(unsigned int i = 1; i < Nx-1; ++i){
        for(unsigned int j = 1; j < Ny-1; ++j){
            ke_exp += abs(psi2d[i][j])*(-0.5*ddxpsi[(i-1)*Ny+(j-1)]);
        }
    }
    ke_exp=ke_exp*dx*dy;

    dcompx tot_energy = (pot_exp + ke_exp);

    delete[] ddxpsi;

    return tot_energy.real();

}


template <class T>
std::string TwoD_TDSE::cstr(T value){
    std::ostringstream out;
    out << value;
    return out.str();
}

int main(){

    TwoD_TDSE twod_tdse;
    twod_tdse.twod_tidse();

    return 0;
}



