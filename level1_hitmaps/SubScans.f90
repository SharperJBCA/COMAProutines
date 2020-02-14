
subroutine SubScanEdges(az, search_length, scan_edges,vars, level, len_az)
  implicit none

  integer, intent(in) :: len_az
  real(8), intent(in) :: az(len_az)
  integer, intent(in) :: search_length
  real(8), intent(in) :: level
  integer, intent(out) :: scan_edges(len_az)
  real(8), intent(out) :: vars(len_az)

  integer :: i

  real(8) :: buffer_values(search_length)
  real(8) :: mean, rms, var, r_search_length

  r_search_length = REAL(search_length)
  
  ! scan_edges is 1 wherever variance of az is < level, 
  !  e.g. azimuth is no moving, therefore end of a scan
  scan_edges(1) = 1
  scan_edges(len_az) = 1
  do i=1, len_az

     ! Fill up the buffer initially
     IF (i < search_length+1) THEN
        buffer_values(i) = az(i)

     ELSE
        ! THEN measure the variance of each step
        buffer_values(MOD(i, search_length)+1) = az(i)
        mean = SUM(buffer_values)/r_search_length
        rms  = SUM(buffer_values*buffer_values)/r_search_length
        var  = SQRT(rms - mean*mean)
        vars(i) =  var
        
        IF (var < level) THEN
           scan_edges(i) = 1
        END IF
     END IF
  END DO
end subroutine SubScanEdges
      
subroutine FindMidPoints(data, edges, len_data)
  implicit none
  
  integer, intent(in) :: len_data
  integer, intent(in) :: data(len_data)
  integer, intent(out) :: edges(len_data)

  integer :: buffer_values(len_data)

  integer :: i, iedge, buffer_count, lastedge, distance
  iedge = 1
  buffer_count = 1

  distance = 2000
  lastedge = 0 - distance

  DO i=1, len_data

     ! Start filling the buffer
     IF (data(i) .EQ. 1) THEN
        buffer_values(buffer_count) = i
        buffer_count = buffer_count + 1
     ELSE ! when you move into data
        IF (buffer_count > 1) THEN ! need to clear the buffer
           if (buffer_values(1) .GT. (lastedge + distance)) THEN
              edges(iedge) = buffer_values(1) !SUM(buffer_values(1:buffer_count))/(buffer_count-1)
              lastedge = edges(iedge)
           end if
           !lastedge = buffer_values(1)
           iedge = iedge + 1
           buffer_count = 1

        END IF
     END IF
  END DO
end subroutine FindMidPoints
