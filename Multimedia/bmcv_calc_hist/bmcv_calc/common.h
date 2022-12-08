

#define ALIGN(x,a) __ALIGN_MASK(x,(typeof(x))(a)-1)

#define __ALIGN_MASK(x,mask) (((x)+(mask))&~(mask))


#define u8 unsigned char 
#define u64 unsigned long long
