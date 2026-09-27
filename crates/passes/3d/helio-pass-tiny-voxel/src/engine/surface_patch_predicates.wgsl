// Exact dyadic comparison for nearly coincident local grid planes. WGSL permits
// float reassociation, so cancellation residuals cannot be geometry authority.
// A numerator (integer +/- f32 fraction), scaled by 2^149, fits in 162 bits
// inside this 256-authored-cell patch. Multiplying a direction's 24-bit
// significand fits in six u32 words. Exponents are compared separately.
alias PatchWide = array<u32,6>;
fn patch_shifted(value:u32,shift:u32)->PatchWide {
    var out:PatchWide;
    let word=shift/32u;let bit=shift%32u;
    out[word]=value<<bit;
    if bit!=0u && word+1u<6u {out[word+1u]=value>>(32u-bit);}
    return out;
}
fn patch_numerator(a:vec2<f32>)->PatchWide {
    var out=patch_shifted(u32(a.x),149u);
    let bits=bitcast<u32>(a.y);
    let exponent=(bits>>23u)&255u;
    let mantissa=(bits&0x7fffffu)|select(0u,0x800000u,exponent!=0u);
    let fraction=patch_shifted(mantissa,u32(max(i32(exponent)-1,0)));
    var carry=0u;
    for(var i=0u;i<6u;i++) {
        if (bits&0x80000000u)!=0u {
            let other=fraction[i]+carry;
            let borrow=u32(other<fraction[i] || out[i]<other);
            out[i]-=other;carry=borrow;
        } else {
            let sum=out[i]+fraction[i];
            let total=sum+carry;
            carry=u32(sum<out[i] || total<sum);out[i]=total;
        }
    }
    return out;
}
fn patch_multiply(a:PatchWide,b:u32)->PatchWide {
    var out:PatchWide;var carry=0u;
    for(var i=0u;i<6u;i++) {
        let al=a[i]&65535u;let ah=a[i]>>16u;
        let bl=b&65535u;let bh=b>>16u;
        let p0=al*bl;let p1=ah*bl;let p2=al*bh;
        let l1=p0+(p1<<16u);
        let l2=l1+(p2<<16u);
        let high=ah*bh+(p1>>16u)+(p2>>16u)+u32(l1<p0)+u32(l2<l1);
        let sum=l2+carry;
        carry=high+u32(sum<l2);out[i]=sum;
    }
    return out;
}
fn patch_leading(a:PatchWide)->i32 {
    for(var i=5;i>=0;i--) {
        if a[u32(i)]!=0u {return i*32+i32(firstLeadingBit(a[u32(i)]));}
    }
    return -1;
}
fn patch_aligned_word(a:PatchWide,shift:u32,word:u32)->u32 {
    let words=shift/32u;let bits=shift%32u;
    if word<words {return 0u;}
    let source=word-words;
    var out=a[source]<<bits;
    if bits!=0u && source>0u {out|=a[source-1u]>>(32u-bits);}
    return out;
}
fn patch_exact_compare(a:vec2<f32>,da:u32,b:vec2<f32>,db:u32)->i32 {
    let ea=(da>>23u)&255u;let eb=(db>>23u)&255u;
    let ma=(da&0x7fffffu)|select(0u,0x800000u,ea!=0u);
    let mb=(db&0x7fffffu)|select(0u,0x800000u,eb!=0u);
    let x=patch_multiply(patch_numerator(a),mb);
    let y=patch_multiply(patch_numerator(b),ma);
    let lx=patch_leading(x);let ly=patch_leading(y);
    if lx<0 {return select(-1,0,ly<0);}
    if ly<0 {return 1;}
    let scale_x=lx+max(i32(eb),1);
    let scale_y=ly+max(i32(ea),1);
    if scale_x<scale_y {return -1;}
    if scale_x>scale_y {return 1;}
    for(var i=5;i>=0;i--) {
        let wx=patch_aligned_word(x,u32(191-lx),u32(i));
        let wy=patch_aligned_word(y,u32(191-ly),u32(i));
        if wx<wy {return -1;}
        if wx>wy {return 1;}
    }
    return 0;
}
fn patch_delta(boundary:i32,anchor:i32,fraction:f32,sign:i32)->vec2<f32> {
    let bits=bitcast<u32>(fraction)^select(0u,0x80000000u,sign>0);
    return vec2<f32>(f32((boundary-anchor)*sign),bitcast<f32>(bits));
}
fn patch_compare(a:vec2<f32>,da:f32,b:vec2<f32>,db:f32)->i32 {
    let ba=bitcast<u32>(da)&0x7fffffffu;let bb=bitcast<u32>(db)&0x7fffffffu;
    if ba==0u {return select(1,0,bb==0u);}
    if bb==0u {return -1;}
    // A generous band encloses rounding of the two sums and products. Very
    // small/subnormal directions go straight to the integer predicate.
    if da>1e-18 && db>1e-18 {
        let x=(a.x+a.y)*db;let y=(b.x+b.y)*da;
        let error=max(abs(x),abs(y))*0.000004;
        if x>1e-30 && y>1e-30 {
            if x<y-error {return -1;}
            if x>y+error {return 1;}
        }
    }
    return patch_exact_compare(a,ba,b,bb);
}
fn patch_distance(a:vec2<f32>,direction:f32)->f32 {
    return ((a.x+a.y)/direction)*0.1;
}
