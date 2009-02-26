PATH=/cygdive/c/Mingw-w64/bin:$PATH
gcc -DRUNTIME=msvcr90 -D__msvcr90__=1 -D__MSVCRT__ -C -E -P -xc-header msvcrt.def.in > msvcr90.def
dlltool --as=as -k --dllname msvcr90.dll --output-lib libmsvcr90.a --def msvcr90.def
for key in printf fprintf sprintf vprintf vfprintf vsprintf; do
	  src=`nm libmsvcr90.a | sed -n -e '/:$/h;/^[0-7][0-7]*  *T  */{s///;H;g;s/\n//p' -e '}' | sed -n 's/:_'"$key"'$//p'`;
	  if test -n "$src"; then
	    dst=`echo "$src" | sed 's/0/4/'`; repl="$repl $dst";
	    tmpfiles="$tmpfiles $src $dst";
	    ar x libmsvcr90.a $src;
	    objcopy --redefine-sym _$key=___msvcrt_$key \
	      --redefine-sym __imp__$key=__imp____msvcrt_$key \
	      $src $dst;
	  fi; 
done;
test `key=_get_output_format; nm libmsvcr90.a | sed -n -e '/:$/h;/^[0-7][0-7]*  *T  */{s///;H;g;s/\n//p' -e '}' | sed -n 's/:_'"$key"'$//p'` || repl="$repl ofmt_stub.o"; 
test -n "$repl" && ar rcs libmsvcr90.a $repl;
rm -f $tmpfiles
