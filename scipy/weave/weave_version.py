major = 0
minor = 4
micro = 0
#release_level = 'alpha'
release_level = ''

if release_level:
    weave_version = '%(major)d.%(minor)d.%(micro)d_%(release_level)s'\
                    % (locals ())
else:
    weave_version = '%(major)d.%(minor)d.%(micro)d'\
                    % (locals ())
