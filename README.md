# polarimetric-slope-sensing
For computing the slope field at the air-water interface given imaging-based measurements of the polarization state of light reflected from that surface.

This project demonstrates the process of computing Stokes parameters from a raw ocean surface image and reconstructing the components as along-look and cross-look directional slope fields.

## Datasets
- A sample raw image of the wave field
- A "lookup table" to obtain _θ_ (angle of incidence) for a given _DoLP_ (degree of linear polarization) value

## Functions
We used three different methods to compute Stokes parameters from the direct measurements of the linearly polarized light intensities at $0^{\circ}$, $45^{\circ}$, $90^{\circ}$, $135^{\circ}$:
1) Bilinear Interpolation of sparse intensity arrays
2) 12-pixel Kernel averaging by Ratliff et al. 2009
3) 12-pixel Kernel Convolution-Demodulation scheme by Ratliff et al. 2009

## Streamline
### Step 1
- Calculation of the polarization orientation ($\phi$) and  ($\theta$) from Stokes parameters

$$\phi=\frac{1}{2} \tan ^{-1}\left(\frac{S_2}{S_1}\right)$$

$$DoLP(\theta, n)=\frac{2 \sin ^2(\theta) \cos (\theta) \sqrt{n^2-\sin ^2(\theta)}}{n^2-\sin ^2(\theta)-n^2 \sin ^2(\theta)+2 \sin ^4(\theta)}$$


### Step 2
- Calculation of the slope fields in the along-look and cross-look direction
  
  $$S_x=-\sin (\phi) \tan (\theta)$$

  $$S_y=-\cos (\phi) \tan (\theta)$$

### Step 3
- Plotting the slope fields in the unit of angles
