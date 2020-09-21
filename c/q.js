    const mem = [1, 2, 3, 40]
   function softmax(ptr, len) {
        let max = mem[ptr]
        for (let i = 1; i < len; ++i)
            max = Math.max(mem[ptr + i], max)
        let sum = 0
        for (let i = 0; i < len; ++i)
            sum += (mem[ptr + i] = Math.exp(mem[ptr + i] - max))
        for (let i = 0; i < len; ++i)
            mem[ptr + i] /= sum
    }


    softmax(0, 4)
    console.log(mem)
