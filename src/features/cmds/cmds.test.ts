import { describe, expect, it } from 'vitest'
import { createcmds } from './cmds'

describe('createcmds', () => {
  it('keeps its contract stable', () => {
    expect(createcmds({ id: 'demo' })).toEqual({ id: 'demo' })
  })
})
