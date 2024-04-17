
import math
import pypim.driver as driver

class Address:

    warps: driver.RangeMask
    rows: driver.RangeMask
    index: int

    def __init__(self, warps, rows, index) -> None:
        super().__init__()
        self.warps = warps
        self.rows = rows
        self.index = index

    def __repr__(self):
        return f'Address: warps = [{self.warps.start}:{self.warps.stop}:{self.warps.step}], rows = [{self.rows.start}:{self.rows.stop}:{self.rows.step}], index = {self.index}'

class MemoryAllocator:
    """
    Manages the memory allocations for the PIM memory.
    """

    def __init__(self, warp_count, warp_cluster_base, warp_size, num_regs) -> None:
        """
        Initializes the memory allocator.
        """

        self.warp_count = warp_count
        self.warp_cluster_base = warp_cluster_base
        self.warp_levels = int(math.log(warp_count, warp_cluster_base))

        self.warp_size = warp_size
        self.num_regs = num_regs

        # Initialize the memory management arrays as free
        self.free_list = [[[True 
                       for _ in range(num_regs)] 
                       for _ in range(self.warp_count // (self.warp_cluster_base ** i))] 
                       for i in range(self.warp_levels + 1)]
        
    def malloc(self, alloc_size, ref_addr = None) -> Address:
        """
        Allocates memory of the given element count.
        """

        # Compute the number of required warps
        alloc_warps = math.ceil(alloc_size / self.warp_size)

        # Round the warp count to the closest power of the warp cluster size
        alloc_level = math.ceil(math.log(alloc_warps, self.warp_cluster_base))
        alloc_warps = self.warp_cluster_base ** alloc_level

        max_elem_per_warp = min(alloc_size, self.warp_size)

        if ref_addr is None:

            # Iterate over the free lists of that level to find an available range
            for curr_idx, curr_free in enumerate(self.free_list[alloc_level]):
                # Find the first free reg
                index = next((i for i, x in enumerate(curr_free) if x), -1)
                # If found free reg
                if index != -1:
                    
                    # Update the allocation tables
                    for sub_level in range(self.warp_levels + 1):
                        start_range = math.floor(curr_idx * (self.warp_cluster_base ** (alloc_level - sub_level)))
                        end_range = math.ceil((curr_idx + 1) * (self.warp_cluster_base ** (alloc_level - sub_level)))
                        for sub_free in self.free_list[sub_level][start_range:end_range]:
                            assert((sub_free[index] == True) or (sub_level > alloc_level))
                            sub_free[index] = False

                    # Return the corresponding allocation range
                    return Address(driver.RangeMask(curr_idx * alloc_warps, (curr_idx + 1) * alloc_warps - 1, 1),
                                driver.RangeMask(0, max_elem_per_warp - 1, 1), 
                                index)
                
        else:

            if (ref_addr.warps.stop - ref_addr.warps.start + 1) != alloc_warps:
                raise RuntimeError(f"Provided reference allocation differs in allocation size.")
            
            alloc_idx = ref_addr.warps.start // alloc_warps

            # Find the first free reg
            index = next((i for i, x in enumerate(self.free_list[alloc_level][alloc_idx]) if x), -1)
            # If found free reg
            if index != -1:
                
                # Update the allocation tables
                for sub_level in range(self.warp_levels + 1):
                    start_range = math.floor(alloc_idx * (self.warp_cluster_base ** (alloc_level - sub_level)))
                    end_range = math.ceil((alloc_idx + 1) * (self.warp_cluster_base ** (alloc_level - sub_level)))
                    for sub_free in self.free_list[sub_level][start_range:end_range]:
                        assert((sub_free[index] == True) or (sub_level > alloc_level))
                        sub_free[index] = False

                # Return the corresponding allocation range
                return Address(driver.RangeMask(alloc_idx * alloc_warps, (alloc_idx + 1) * alloc_warps - 1, 1),
                            driver.RangeMask(0, max_elem_per_warp - 1, 1), 
                            index)

        raise RuntimeError(f"Out of memory: failed to allocate {alloc_size} elements.")
    
    def free(self, address: Address):
        """
        Frees the given crossbar range
        """

        warp_range = address.warps
        index = address.index

        alloc_warps = warp_range.stop - warp_range.start + 1
        alloc_level = int(math.log(alloc_warps, self.warp_cluster_base))
        alloc_idx = warp_range.start // alloc_warps

        # Update the free lists
        for level in range(self.warp_levels + 1):
            start_range = math.floor(alloc_idx * (self.warp_cluster_base ** (alloc_level - level)))
            end_range = math.ceil((alloc_idx + 1) * (self.warp_cluster_base ** (alloc_level - level)))

            for i, sub_free in enumerate(self.free_list[level][start_range:end_range]):
                assert(not sub_free[index])

                # Levels up to the allocation level automatically become free
                if level <= alloc_level:
                    sub_free[index] = True

                # Levels above the allocation level become the AND of their children
                else:
                    sub_free[index] = all([x[index] for x in self.free_list[level - 1]
                        [(start_range + i) * self.warp_cluster_base: (start_range + i + 1) * self.warp_cluster_base]])

# The default memory allocator
alloc = MemoryAllocator(driver.numWarps(), driver.warpClusterBase(), driver.warpSize(), driver.numRegs())

def malloc(alloc_size, ref_addr = None) -> Address:
    return alloc.malloc(alloc_size, ref_addr)

def free(address: Address):
    alloc.free(address)
