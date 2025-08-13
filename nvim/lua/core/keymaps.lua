-- For conciseness
local opts = {noremap = true, silent = true}

-- save file
vim.keymap.set('n', '<C-s>', '<cmd> w <CR>', opts)

-- delete single character without copying into register
vim.keymap.set('n', 'x', '"_x', opts)
