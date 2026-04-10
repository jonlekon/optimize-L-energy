#-------------------- Detect Closed Algebraic Expressions --------------------
module Detect_Closed_Expressions
    
using Symbolics
using LinearAlgebra

export detect_matrix, detect_number 


# detect rational multiples / rational powers of known constants
function detect_transcendental_linear_power(x; tol=1e-7, maxden=20, maxk=5, precision=256)
    xb = BigFloat(x, precision)
    
    # Known constants
    constants = Dict(
        "π" => BigFloat(pi, precision),
        "e" => BigFloat(ℯ, precision),
        "φ" => (1 + sqrt(BigFloat(5, precision))) / 2,
        "γ" => BigFloat(0.57721566490153286060, precision)  # Euler-Mascheroni constant
    )
    
    for (name, value) in constants
        # --- Rational multiples ---
        ratio = xb / value
        r = try
            rationalize(ratio; tol=tol)
        catch
            nothing
        end
        if r !== nothing && denominator(r) <= maxden
            # Compute numeric value
            numeric_val = BigFloat(r) * value
            # Construct symbolic string
            symbolic_str = r == 1//1 ? name : "$(r)*$name"
            return (numeric_val, symbolic_str)
        end

        # --- Rational powers ---
        for k in 2:maxk
            y = xb^k / value
            r2 = try
                rationalize(y; tol=tol)
            catch
                nothing
            end
            if r2 !== nothing && denominator(r2) <= maxden
                # Compute numeric value
                numeric_val = (value * BigFloat(r2))^(1//k)
                symbolic_str = "$(r2)*$name^(1//$k)"
                return (numeric_val, symbolic_str)
            end
        end
    end

    return nothing
end


function detect_rational_power(x;
        c = 1.0,
        max_a = 10,
        max_b = 10,
        max_p = 10,
        max_q = 10,
        tol = 1e-6)

    # Store original sign and work with absolute value
    s = sign(x)
    abs_x = abs(x / c)

    best = nothing
    best_err = Inf

    # Search over denominators q and numerators p
    for q in 1:max_q
        for p in 1:max_p
            # compute candidate root
            root = abs_x^(q / p)

            for b in 1:max_b
                a = round(Int, root * b)
                a == 0 && continue

                # candidate base
                base = a / b
                val_pos = base^(p / q)        # always positive
                val = s * val_pos             # restore original sign

                err = abs(abs_x * c * s - val * c)  # absolute error in original scale

                if err < tol && err < best_err
                    best_err = err
                    best = (a, b, p, q, c, val)
                end
            end
        end
    end

    best === nothing && return nothing

    a, b, p, q, c, val = best

    # construct symbolic string
    base_str = "($(a//b))"
    pow_str  = q == 1 ? "$base_str^$p" : "$base_str^($p//$q)"
    sym = s < 0 ? "-$pow_str" : pow_str
    sym = c == 1 ? sym : "$c * $sym"

    return (
        value = BigFloat(c * val),
        symbolic = sym,
        error = best_err
    )
end

# Detect small-denominator rationals
function detect_rational(x; tol=1e-7, maxden=20, precision=256)
    xb = BigFloat(x, precision)

    # Try to rationalize the number
    r = try
        rationalize(xb; tol=tol)
    catch
        nothing
    end

    if r !== nothing && denominator(r) <= maxden
        # Compute precise numeric value
        numeric_val = BigFloat(r)
        # Construct symbolic string
        symbolic_str = denominator(r) == 1 ? string(numerator(r)) : string(r)
        return (numeric_val, symbolic_str)
    else
        return nothing
    end
end

function detect_number(x; tol=1e-7, precision=50)
    if(x<1e-6 && x>-1e-6)
        return (0.0, "0")
    end
    xb = x #BigFloat(x, precision)
    candidates = []

    # 1. Transcendental linear / power
   # t = detect_transcendental_linear_power(xb, tol=tol)
   # if t !== nothing
   #     numeric_val, symbolic_str = t
   #     error_val = abs(xb - numeric_val)
   #     push!(candidates, (error_val, numeric_val, symbolic_str))
   # end

    # 2. Rational (small denominator)
    r = detect_rational(xb, tol=tol)
    if r !== nothing
        numeric_val, symbolic_str = r
        error_val = abs(xb - numeric_val)
        push!(candidates, (error_val, numeric_val, symbolic_str))
    end

    # 3. Rational power / radical
    p = detect_rational_power(xb, tol=tol)
    if p !== nothing
        numeric_val, symbolic_str = p
        error_val = abs(xb - numeric_val)
        push!(candidates, (error_val, numeric_val, symbolic_str))
    end

    # --- fallback numeric only if no candidates found ---
    if isempty(candidates)
        return (xb, string(xb))
    end

    # --- select candidate with smallest error ---
    best_candidate = reduce((a, b) -> a[1] <= b[1] ? a : b, candidates)

    # safely extract numeric_val and symbolic_str
    best_val = best_candidate[2]
    best_str = best_candidate[3]

    return (best_val, best_str)
end

function detect_matrix(A::AbstractMatrix; tol=1e-7)
    if tol < 1e-6
        tol = 1e-6
    end
    n, m = size(A)
    numeric_mat = zeros(Float64, n, m)
    symbolic_mat = Matrix{Any}(undef, n, m)

    for i in 1:n, j in 1:m
        num_val, sym_val = detect_number(A[i,j];tol=tol)
        numeric_mat[i,j] = num_val
        symbolic_mat[i,j] = sym_val
    end
 # println(det(numeric_mat))
    return (
    numeric  = numeric_mat,
    symbolic = symbolic_mat
    )
end


end # module
