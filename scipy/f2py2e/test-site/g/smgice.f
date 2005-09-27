       subroutine ICEIN(iafoil,xa,ya,filename) 

       include 'param.in'
       real xa(ia), ya(ia)
       character*20 filename
      
       open(1,file=filename)
       write(1,*) iafoil
       do i=1,iafoil
       write(1,*) xa(i), ya(i)
       enddo
       close(1)

       return
       end
