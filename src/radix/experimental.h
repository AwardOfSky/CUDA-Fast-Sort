/*
    Experimental header

    Features:
        Reflection Prototyping
        (To be used to support AoS API calls when NVCC gets reflection)
*/
#pragma once


// check for reflection capabilities
#if !defined(__CUDACC__) && \
    defined(__cpp_impl_reflection) && (__cpp_impl_reflection >= 202506L)
    #define RSORT_CPP_REFLECTION 1
#else
    #define RSORT_CPP_REFLECTION 0
#endif


// experimental C++ reflection prototype (tested with g++-16 - 16.0.1 20260315)
#if RSORT_CPP_REFLECTION

#include <cstdlib>
#include <cstdint>
#include <typeinfo>
#include <meta>


namespace rsort {

// quick fixed_string implementation
template<size_t N>
struct fixed_string {
    char data[N];

    // initializers 
    constexpr fixed_string(const char (&s)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = s[i];
        }
    }

    // operators
    constexpr bool operator==(const fixed_string&) const = default;
    
    constexpr char operator[](size_t i) const {
        return data[i];
    }

    // std functions
    constexpr const char* c_str() const {
        return data;
    }

    constexpr std::size_t size() const {
        return N ? (N - 1) : 0;
    }

    constexpr std::string_view view() const {
        return {data, size()};
    }
};


// ---- reflection helpers ----

// find member in type, return reflection info
template<typename T, fixed_string Name>
consteval std::meta::info refl_get_member() {
    constexpr auto ctx = std::meta::access_context::current();

    for (auto m : std::meta::nonstatic_data_members_of(^^T, ctx)) {
        if (std::meta::identifier_of(m) == Name.view()) {
            return m;
        }
    }

    return std::meta::info{};
}

// return reflection info of all members of type except first "member"
template<typename T, fixed_string Name>
consteval std::vector<std::meta::info> refl_getallbut_member() {
    constexpr auto ctx = std::meta::access_context::current();
    std::vector<std::meta::info> data_fields;        

    for (std::meta::info m : std::meta::nonstatic_data_members_of(^^T, ctx)) {
        if (std::meta::identifier_of(m) != Name.view()) {   
            data_fields.push_back(m);
        }
    }

    return data_fields;
}

// return reflected data for a member with type and identifier m
consteval std::meta::info refl_meta_to_member(std::meta::info m) {
    return std::meta::data_member_spec(
        std::meta::type_of(m),
        {.name = std::meta::identifier_of(m)}
    );
}

// transform array of metadata into members, according to refl_meta_to_member
consteval std::vector<std::meta::info>
refl_metavec_to_members(std::vector<std::meta::info> arr) {
    for (std::meta::info& m : arr) {
        m = refl_meta_to_member(m);
    }

    return arr;
}

template<typename T>
using refl_ptr_t = T*;

// return reflection of a pointer to type
template<typename T>
consteval std::meta::info refl_ptr() {
    return ^^refl_ptr_t<T>;
}

// reflect metadata into a C++ type
template<std::meta::info meta>
using refl_meta_to_type = typename [: std::meta::type_of(meta) :];

// reflect metadata of a type into a C++ type
template<std::meta::info meta>
using refl_metatype_to_type = typename [: meta :];


/*
    Transform AoS:
        KV Pair  [Array]
            key
            member_a
            member_b
            member_c
            ...
    
    Into SoA:
        SoA Type
            key [array]
            value [array]
                member_a
                member_b
                member_c
                ...   
*/
template<typename Pair, fixed_string Key>
struct soa_traits {
    struct value_type;
    struct soa_type;


    static consteval std::meta::info get_key() {
        return refl_get_member<Pair, Key>();
    }

    consteval {
        auto value_fields = refl_metavec_to_members(refl_getallbut_member<Pair, Key>());
        
        constexpr std::meta::info key_meta = get_key();
        static_assert(
            key_meta != std::meta::info{},
            "[soa_traits]: key name not found in type!"
        );

        std::meta::define_aggregate(^^value_type, value_fields);

        std::vector<std::meta::info> soa_fields = {
            std::meta::data_member_spec(
                refl_ptr<value_type>(),
                {.name = "value"}
            ),
            std::meta::data_member_spec(
                refl_ptr<refl_meta_to_type<key_meta>>(),
                {.name = std::meta::identifier_of(key_meta)}
            )
        };

        std::meta::define_aggregate(^^soa_type, soa_fields);
    }
};

} // namespace rsort

#endif // RSORT_CPP_REFLECTION
