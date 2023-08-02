if not is_variable('CPU_MTARGETS_DEBUG')
  error('Variable "CPU_MTARGETS_DEBUG" is not defined. CPU dispatcher is not properly initialized.')
endif

targets_info = {}
targets_config = {}
foreach config_name, mtargets_info : CPU_MTARGETS_DEBUG
  foreach target_name, target_info : mtargets_info
    if target_name not in targets_info
      targets_info += {target_name: target_info}
    endif
    if target_name not in targets_config
      targets_config += {target_name: []}
    endif
    targets_config += {
      target_name: targets_config[target_name] + [config_name]
    }
  endforeach
endforeach

target_info_tpl = \
'''
  Target      : @0@
  Enabled     : @1@
  Flags       : @2@
  Detect      : @3@
  Defines     : @4@
  Undefines   : @5@
  Sources     : @6@
'''
generated_targets = ''
foreach target_name, target_info : targets_info
  generated_targets += target_info_tpl.format(
    target_name,
    ' '.join(target_info['features']),
    ' '.join(target_info['args']),
    ' '.join(target_info['detect']),
    ' '.join(target_info['defines']),
    ' '.join(target_info['undefines']),
    '\n                '.join(targets_config[target_name])
  )
endforeach
debug('Generated CPU Targets:', generated_targets)
