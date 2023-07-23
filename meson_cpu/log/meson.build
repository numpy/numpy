foreach cpu_req_var : [
  'CPU_CONF_BASELINE', 'CPU_BASELINE_NAMES',
  'CPU_CONF_DISPATCH', 'CPU_DISPATCH_NAMES',
  'CPU_MTARGETS_LOG',
]
  if not is_variable(cpu_req_var)
    error('Variable', cpu_req_var, 'is not exist. CPU dispatcher is not properly initialized.')
  endif
endforeach

targets_info = {}
targets_config = {}
foreach config_name, mtargets_info : CPU_MTARGETS_LOG
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

message(
'''
########### CPU OPTIMIZATION ###########
CPU baseline  :
  Requested   : @0@
  Enabled     : @1@
CPU dispatch  :
  Requested   : @2@
  Enabled     : @3@
Generated     : @4@
'''.format(
    CPU_CONF_BASELINE, ' '.join(CPU_BASELINE_NAMES),
    CPU_CONF_DISPATCH, ' '.join(CPU_DISPATCH_NAMES),
    generated_targets
  )
)
