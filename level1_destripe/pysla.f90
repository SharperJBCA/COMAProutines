!!Quick routine to wrap the SLALIB moon position function.
!!It is inaccurate but fine for QUIJOTE accuracy. It downsamples
!! the data by 100x!

subroutine refro(zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref, len_zd)
  implicit none
  integer, intent(in) :: len_zd
  real*8, intent(in) :: zd(len_zd)
  real*8, intent(out) :: ref(len_zd)
  real*8, intent(in) :: hm ! height
  real*8, intent(in) :: tdk ! ambient temp (K)
  real*8, intent(in) :: pmb ! pressure (mb)
  real*8, intent(in) :: rh ! relative humidity (0-1)
  real*8, intent(in) :: wl ! effective wavelength (um)
  real*8, intent(in) :: tlr ! latitude of observer
  real*8, intent(in) :: phi ! temperature lapse rate (K/m)
  real*8, intent(in) :: eps ! precision required (radian)

  !f2py real*8 zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref
  !f2py integer len_zd

  integer :: i

  do i=1, len_zd
     call sla_refro(zd(i), hm, tdk, pmb, rh, wl, phi, tlr, eps, ref(i))
  enddo


end subroutine refro

subroutine rdplan(jd, np, lon, lat, ra, dec, diam, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: diam(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,jd, diam

  !integer :: i

  real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step


  do k=1, len_bn
     call sla_rdplan(jd(k),np,lon*pi/180.0,lat*pi/180.0,ra(k),dec(k),diam(k))
  enddo
  
end subroutine rdplan




subroutine planet(jd, np, dist, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: dist(6,len_bn)

  !f2py integer len_bn
  !f2py real*8 jd, dist

  integer :: jstat

  !real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step

  !mask = 1.0d0

  do k=1, len_bn
     call sla_planet(jd(k),np,dist(:,k),jstat)
  enddo
  
end subroutine planet


subroutine h2e(az, el, mjd, lon, lat, ra, dec, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: az(len_bn)
  real*8, intent(in) :: el(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: gmst

  do i=1, len_bn
     call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     gmst = sla_gmst(mjd(i))
     ra(i) = gmst + lon - ra(i)
     ra(i) = sla_dranrm(ra(i))
  enddo    

  
end subroutine h2e

subroutine e2h(ra, dec, mjd, lon, lat, az, el,lha, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: az(len_bn)
  real*8, intent(out) :: el(len_bn)
  real*8, intent(out) :: lha(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface


  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 az,el,mjd

  integer :: i
  real*8 :: gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     lha(i) = lon + gmst - ra(i) ! CONVERT TO LHA

     call sla_de2h(lha(i), dec(i), lat, az(i), el(i))
  enddo    

  
end subroutine e2h

subroutine gmst(mjd, gmst_out, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: gmst_out(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  integer :: i

  do i=1, len_bn
     gmst_out(i) = sla_gmst(mjd(i))
  enddo
end subroutine gmst

subroutine precess(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', epoch, 2000D0, ra(i), dec(i))
  enddo    
  
end subroutine precess


subroutine precess_year(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', 2000D0, epoch, ra(i), dec(i))
  enddo    
  
end subroutine precess_year





subroutine pa(ra, dec,mjd, lon,lat,pang, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: pang(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_pa(radummy, decdummy, phidummy)
     real*8 :: radummy, decdummy, phidummy
     END FUNCTION sla_pa
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd, lon, lat
  !f2py real*8 ra,dec, pang

  integer :: i
  real*8 :: ha, gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     ha = gmst + lon - ra(i)
     pang(i) = sla_pa(ha, dec(i), lat)
  enddo    

  
end subroutine pa


subroutine e2g(ra, dec, gl, gb, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: gl(len_bn)
  real*8, intent(out) :: gb(len_bn)

  !f2py integer len_bn
  !f2py real*8 ra,dec,gl,gb

  integer :: i


  do i=1, len_bn
     call sla_eqgal(ra(i), dec(i), gl(i), gb(i))
  enddo    

  
end subroutine e2g

! module calc_source_position

! contains 

! function source_ang_sep(phi0,theta0, phi1, theta1) result(sep)
!   real*8, intent(in) :: phi0
!   real*8, intent(in) :: theta0
!   real*8, intent(in) :: phi1
!   real*8, intent(in) :: theta1

!   real*8, intent(out) :: sep
!   real*8 :: A

!   A = sin(theta0) * sin(theta1) + cos(theta0) * cos(theta1) * cos(phi0-phi1)
!   sep = acos(A)
! end function source_ang_sep

! function to_vector(phi, theta) result(vec)

!   real*8, intent(in) :: phi
!   real*8, intent(in) :: theta

!   real*8, intent(out) :: vec(3)

!   vec(1) = cos(phi) * cos(theta)
!   vec(2) = sin(phi) * cos(theta)
!   vec(3) = sin(theta)

! end function to_vector

subroutine source_target(width, ha0, ra0, dec0,mjd0, lon, lat, az_mid, el_mid, mjd_start,mjd_delta)
  implicit none

  real*8, intent(in) :: width
  real*8, intent(in) :: ha0
  real*8, intent(in) :: ra0
  real*8, intent(in) :: dec0
  real*8, intent(in) :: mjd0
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  
  real*8, intent(out) :: az_mid
  real*8, intent(out) :: el_mid
  real*8, intent(out) :: mjd_start
  real*8, intent(out) :: mjd_delta

  !f2py intent(out) az_mid
  !f2py intent(out) el_mid
  !f2py intent(out) mjd_start
  !f2py intent(out) mjd_delta

  real*8 :: stepsize
  real*8 :: ha_min_start, ha_min_end, ha_start, ha_end, ha
  real*8 :: mjd, mjd_end, mjd_mid, gmst, az, el
  integer :: i, istart, iend, steps

  real*8 :: PI

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  PI = 4.D0*datan(1.D0)

  istart = 0
  iend = 0
  steps = 24 !85564
  stepsize = 1.0/24.0 !86400.0 ! 1 second steps
  
  mjd = mjd0
  ha_start = ha0-width/2.0/cos(dec0)
  ha_end   = ha0+width/2.0/cos(dec0)

  ha_min_start = 1e6 
  ha_min_end   = 1e6
  do i=1, steps
     
     ! calculate the az/el of the source
     gmst = sla_gmst(mjd0 + stepsize*(i-1))
     ha = modulo(gmst + lon -ra0, 2.D0*PI) !- ra0
     call sla_de2h(ha, dec0, lat, az, el)

     if (abs(ha-ha_start) < ha_min_start .AND. el > 0) then
        istart = i
        ha_min_start = abs(ha-ha_start)
     end if 
        
     if (abs(ha-ha_end) < ha_min_end  .AND. el > 0) then
        iend = i
        ha_min_end = abs(ha-ha_end)
     end if 
     !if (i == 1) then
     !   print *, ha_min_end , abs(ha-ha_end), ha_min_start, abs(ha-ha_start), el
     !end if
     print *, ha_min_end , abs(ha-ha_end), ha_min_start, abs(ha-ha_start), el

  end do

  do i=istart, iend, 100
     
     ! calculate the az/el of the source
     gmst = sla_gmst(mjd0 + stepsize*(i-1))
     ha = modulo(gmst + lon -ra0, 2.D0*PI) !- ra0
     call sla_de2h(ha, dec0, lat, az, el)

     if (abs(ha-ha_start) < ha_min_start .AND. el > 0) then
        istart = i
        ha_min_start = abs(ha-ha_start)
     end if 
        
     if (abs(ha-ha_end) < ha_min_end  .AND. el > 0) then
        iend = i
        ha_min_end = abs(ha-ha_end)
     end if 
     print *, ha_min_end , abs(ha-ha_end), ha_min_start, abs(ha-ha_start)
  end do
  !print *, 'HELLO', istart, iend, ha_min_start, ha_min_end, width/cos(dec0)

  mjd_start = mjd0 + stepsize*(istart-1)
  mjd_end   = mjd0 + stepsize*(iend-1)
  print *, istart, iend, mjd_start,mjd_end


  print*, '---'


  if (mjd_start > mjd_end) then
     mjd_end = mjd_end + 0.9903241 ! sidereal day
  end if

  mjd_mid   = (mjd_start + mjd_end)/2.0
  mjd_delta = (mjd_end - mjd_start)

  gmst = sla_gmst(mjd_mid)
  ha = gmst + lon - ra0
  call sla_de2h(ha, dec0, lat, az_mid, el_mid)
    
end subroutine source_target

! end module calc_source_position
