#!/bin/sh
<<COMMENT
      For every c.src file, this script will find the corresponding
      .c file (or if its a dispatch file, then then take .dispatch.<?>.c)
      and add it to gitignore, if not already done.
      Same for .h.src files.

      Note: This will not find all the gitignorable files
      as some are generated with python and not the src file.
COMMENT

declare -a arr=()
cd $(git rev-parse --show-toplevel)

find_ignorables() {

    # Find files containing the .src extension
    for src_path in $(find '.'  -name "*.$1.src" -printf '%P\n' | \
        sed "s/\.$1.src//;");
    do
        cur_dir=$(dirname "$src_path")
        filename=$(basename "$src_path")

        # Look in the same directory as the .src file do we see any
        # other file in the same dir, with the same name, without the
        # .src?
        for full_name in $(find "$cur_dir" -name "$filename.$1" \
            -printf '%P\n');
        do
            if ! git check-ignore -q "$cur_dir/$full_name"; then
                arr+=("$cur_dir/$full_name");
            fi
        done

        # Dispatch files sometimes have <name>.dispatch.<crazy>.c get
        # these too
        if echo $filename | grep -q ".*\.dispatch"; then
            for full_name in $(find "$cur_dir" -name "$filename.*.$1" \
                -printf '%P\n');
            do
                if ! git check-ignore -q "$cur_dir/$full_name"; then
                    arr+=("$cur_dir/$full_name");
                fi
            done
        fi

    done
}

find_ignorables 'c'
find_ignorables 'h'

echo "The following files are eligible for .gitignore and have" \
"been added as comments:"
printf "%s\n" "${arr[@]}"
printf "#Recommended by gitignoreFinder.sh:\n" >> '.gitignore'
printf "#%s\n" "${arr[@]}" >> '.gitignore'