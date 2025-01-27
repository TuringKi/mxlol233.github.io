---
title: gcc/cp parser.cc call graph
date: 2023-1-20
permalink: /gcc-cp-parser-call-graph
---

This page shows the call-graph in gcc/cp/parser.cc.


```c++
── cp_parser_translation_unit(cp_parser *)/
    ├── make_declarator(cp_declarator_kind)/
    │   └── alloc_declarator(size_t)
    ├── make_parameter_declarator(cp_decl_specifier_seq *, cp_declarator *, tree, location_t, bool)
    ├── cp_lexer_peek_token(cp_lexer *)/
    │   └── cp_lexer_print_token(FILE *, cp_token *)/
    │       └── tree_check(tree, const char *, int, const char *, tree_code)
    ├── cp_lexer_consume_token(cp_lexer *)
    ├── cp_parser_module_declaration(cp_parser *, module_parse, bool)/
    │   ├── cp_lexer_next_token_is(cp_lexer *, enum cpp_ttype)
    │   ├── cp_parser_require_pragma_eol(cp_parser *, cp_token *)/
    │   │   ├── cp_parser_require(cp_parser *, enum cpp_ttype, required_token, location_t)/
    │   │   │   ├── cp_parser_simulate_error(cp_parser *)/
    │   │   │   │   └── cp_parser_uncommitted_to_tentative_parse_p(cp_parser *)
    │   │   │   └── cp_parser_required_error(cp_parser *, required_token, bool, location_t)/
    │   │   │       └── cp_parser_error_1(cp_parser *, const char *, required_token, location_t)/
    │   │   │           ├── cp_parser_skip_to_pragma_eol(cp_parser *, cp_token *)/
    │   │   │           │   └── cp_lexer_next_token_is_not(cp_lexer *, enum cpp_ttype)
    │   │   │           ├── cp_lexer_peek_conflict_marker(cp_lexer *, enum cpp_ttype, location_t *)/
    │   │   │           │   ├── cp_lexer_peek_nth_token(cp_lexer *, size_t)
    │   │   │           │   └── get_finish(location_t)
    │   │   │           ├── cp_lexer_safe_previous_token(cp_lexer *)/
    │   │   │           │   └── cp_lexer_previous_token_position(cp_lexer *)
    │   │   │           ├── cp_parser_is_string_literal(cp_token *)
    │   │   │           └── get_matching_symbol(required_token)
    │   │   └── cp_parser_skip_to_pragma_eol(cp_parser *, cp_token *)
    │   ├── cp_lexer_nth_token_is_keyword(cp_lexer *, size_t, enum rid)/
    │   │   └── cp_lexer_peek_nth_token(cp_lexer *, size_t)
    │   ├── cp_lexer_nth_token_is(cp_lexer *, size_t, enum cpp_ttype)
    │   ├── cp_parser_module_name(cp_parser *)/
    │   │   └── cp_parser_error(cp_parser *, const char *)/
    │   │       └── cp_parser_simulate_error(cp_parser *)
    │   └── cp_parser_attributes_opt(cp_parser *)/
    │       ├── cp_next_tokens_can_be_gnu_attribute_p(cp_parser *)/
    │       │   └── cp_nth_tokens_can_be_gnu_attribute_p(cp_parser *, size_t)
    │       ├── attr_chainon(tree, tree)
    │       ├── cp_parser_gnu_attributes_opt(cp_parser *)/
    │       │   ├── token_pair::require_open(cp_parser *)
    │       │   └── cp_parser_gnu_attribute_list(cp_parser *, bool)/
    │       │       ├── canonicalize_attr_name(tree)/
    │       │       │   └── tree_check(tree, const char *, int, const char *, tree_code)
    │       │       ├── is_attribute_p(const char *, const_tree)/
    │       │       │   ├── tree_check(const_tree, const char *, int, const char *, tree_code)
    │       │       │   └── tree_check(const_tree, const char *, int, const char *, tree_code)
    │       │       └── cp_parser_parenthesized_expression_list(cp_parser *, int, bool, bool, bool *, location_t *, bool)/
    │       │           ├── cp_parser_conditional_expression(cp_parser *)/
    │       │           │   └── cp_parser_binary_expression(cp_parser *, bool, bool, bool, enum cp_parser_prec, cp_id_kind *)/
    │       │           │       ├── cp_parser_cast_expression(cp_parser *, bool, bool, bool, cp_id_kind *)/
    │       │           │       │   ├── cp_parser_parse_tentatively(cp_parser *)/
    │       │           │       │   │   ├── cp_parser_context_new(cp_parser_context *)
    │       │           │       │   │   └── cp_lexer_save_tokens(cp_lexer *)
    │       │           │       │   ├── token_pair::consume_open(cp_parser *)
    │       │           │       │   ├── cp_parser_skip_to_closing_parenthesis(cp_parser *, bool, bool, bool)/
    │       │           │       │   │   └── cp_parser_skip_to_closing_parenthesis_1(cp_parser *, bool, cpp_ttype, bool)
    │       │           │       │   ├── cp_parser_tokens_start_cast_expression(cp_parser *)
    │       │           │       │   ├── cp_lexer_rollback_tokens(cp_lexer *)/
    │       │           │       │   │   └── vec::pop()
    │       │           │       │   ├── cp_parser_type_id(cp_parser *, cp_parser_flags, location_t *)/
    │       │           │       │   │   └── cp_parser_type_id_1(cp_parser *, cp_parser_flags, bool, bool, location_t *)/
    │       │           │       │   │       └── cp_parser_type_specifier_seq(cp_parser *, cp_parser_flags, bool, bool, cp_decl_specifier_seq *)/
    │       │           │       │   │           ├── cp_next_tokens_can_be_attribute_p(cp_parser *)/
    │       │           │       │   │           │   └── cp_next_tokens_can_be_std_attribute_p(cp_parser *)/
    │       │           │       │   │           │       └── cp_nth_tokens_can_be_std_attribute_p(cp_parser *, size_t)
    │       │           │       │   │           ├── cp_next_tokens_can_be_gnu_attribute_p(cp_parser *)
    │       │           │       │   │           ├── cp_parser_skip_attributes_opt(cp_parser *, size_t)/
    │       │           │       │   │           │   ├── cp_nth_tokens_can_be_gnu_attribute_p(cp_parser *, size_t)
    │       │           │       │   │           │   ├── cp_parser_skip_gnu_attributes_opt(cp_parser *, size_t)/
    │       │           │       │   │           │   │   └── cp_parser_skip_balanced_tokens(cp_parser *, size_t)
    │       │           │       │   │           │   └── cp_parser_skip_std_attribute_spec_seq(cp_parser *, size_t)
    │       │           │       │   │           └── cp_parser_type_specifier(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, bool, int *, bool *)/
    │       │           │       │   │               ├── cp_parser_enum_specifier(cp_parser *)/
    │       │           │       │   │               │   ├── cp_lexer_next_token_is_keyword(cp_lexer *, enum rid)
    │       │           │       │   │               │   ├── cp_parser_nested_name_specifier_opt(cp_parser *, bool, bool, bool, bool, bool)/
    │       │           │       │   │               │   │   ├── cp_parser_pre_parsed_nested_name_specifier(cp_parser *)
    │       │           │       │   │               │   │   ├── cp_parser_nth_token_starts_template_argument_list_p(cp_parser *, size_t)
    │       │           │       │   │               │   │   ├── cp_parser_optional_template_keyword(cp_parser *)
    │       │           │       │   │               │   │   ├── cp_parser_qualifying_entity(cp_parser *, bool, bool, bool, bool, bool)
    │       │           │       │   │               │   │   ├── tree_operand_check(tree, int, const char *, int, const char *)
    │       │           │       │   │               │   │   ├── ovl_first(tree)
    │       │           │       │   │               │   │   ├── function_concept_p(tree)
    │       │           │       │   │               │   │   ├── variable_concept_p(tree)
    │       │           │       │   │               │   │   ├── standard_concept_p(tree)
    │       │           │       │   │               │   │   ├── variable_template_p(tree)
    │       │           │       │   │               │   │   ├── cp_parser_lookup_name(cp_parser *, tree, enum tag_types, bool, bool, bool, tree *, location_t)
    │       │           │       │   │               │   │   ├── cp_parser_name_lookup_error(cp_parser *, tree, tree, name_lookup_error, location_t)
    │       │           │       │   │               │   │   ├── check_template_keyword_in_nested_name_spec(tree)
    │       │           │       │   │               │   │   └── cp_lexer_purge_tokens_after(cp_lexer *, cp_token *)
    │       │           │       │   │               │   ├── cp_parser_identifier(cp_parser *)
    │       │           │       │   │               │   ├── cp_parser_commit_to_tentative_parse(cp_parser *)/
    │       │           │       │   │               │   │   ├── cp_lexer_saving_tokens(const cp_lexer *)
    │       │           │       │   │               │   │   └── cp_lexer_commit_tokens(cp_lexer *)
    │       │           │       │   │               │   ├── cp_parser_check_type_definition(cp_parser *)
    │       │           │       │   │               │   ├── token_pair::consume_open(cp_parser *)
    │       │           │       │   │               │   ├── cp_parser_skip_to_end_of_block_or_statement(cp_parser *)/
    │       │           │       │   │               │   │   └── abort_fully_implicit_template(cp_parser *)
    │       │           │       │   │               │   └── cp_parser_enumerator_list(cp_parser *, tree)/
    │       │           │       │   │               │       └── cp_parser_enumerator_definition(cp_parser *, tree)
    │       │           │       │   │               ├── cp_parser_set_decl_spec_type(cp_decl_specifier_seq *, tree, cp_token *, bool)/
    │       │           │       │   │               │   ├── decl_spec_seq_has_spec_p(const cp_decl_specifier_seq *, cp_decl_spec)
    │       │           │       │   │               │   └── set_and_check_decl_spec_loc(cp_decl_specifier_seq *, cp_decl_spec, cp_token *)/
    │       │           │       │   │               │       └── token_is__thread(cp_token *)
    │       │           │       │   │               ├── cp_parser_class_specifier(cp_parser *)/
    │       │           │       │   │               │   ├── cp_parser_class_head(cp_parser *, bool *)/
    │       │           │       │   │               │   │   ├── cp_parser_class_key(cp_parser *)
    │       │           │       │   │               │   │   ├── find_contract(tree)
    │       │           │       │   │               │   │   ├── cp_parser_global_scope_opt(cp_parser *, bool)
    │       │           │       │   │               │   │   ├── cp_parser_class_name(cp_parser *, bool, bool, enum tag_types, bool, bool, bool, bool)
    │       │           │       │   │               │   │   ├── cp_parser_check_for_invalid_template_id(cp_parser *, tree, enum tag_types, location_t)
    │       │           │       │   │               │   │   ├── cp_parser_virt_specifier_seq_opt(cp_parser *)
    │       │           │       │   │               │   │   ├── cp_parser_next_token_starts_class_definition_p(cp_parser *)
    │       │           │       │   │               │   │   ├── cp_parser_check_template_parameters(cp_parser *, unsigned int, bool, location_t, cp_declarator *)
    │       │           │       │   │               │   │   ├── tree_int_cst_elt_check(tree, int, const char *, int, const char *)
    │       │           │       │   │               │   │   └── cp_parser_base_clause(cp_parser *)
    │       │           │       │   │               │   ├── token_pair::require_open(cp_parser *)
    │       │           │       │   │               │   ├── cp_ensure_no_omp_declare_simd(cp_parser *)
    │       │           │       │   │               │   ├── cp_ensure_no_oacc_routine(cp_parser *)
    │       │           │       │   │               │   ├── cp_parser_skip_to_closing_brace(cp_parser *)
    │       │           │       │   │               │   ├── cp_parser_member_specification_opt(cp_parser *)/
    │       │           │       │   │               │   │   ├── cp_parser_pragma(cp_parser *, enum pragma_context, bool *)
    │       │           │       │   │               │   │   └── cp_parser_member_declaration(cp_parser *)
    │       │           │       │   │               │   ├── cp_lexer_set_token_position(cp_lexer *, cp_token_position)
    │       │           │       │   │               │   ├── cp_parser_late_parsing_default_args(cp_parser *, tree)/
    │       │           │       │   │               │   │   ├── push_unparsed_function_queues(cp_parser *)
    │       │           │       │   │               │   │   ├── vec_safe_push(releasing_vec &, const tree &)
    │       │           │       │   │               │   │   ├── tree_check2(tree, const char *, int, const char *, enum tree_code, enum tree_code)
    │       │           │       │   │               │   │   ├── releasing_vec::operator[](ptrdiff_t)
    │       │           │       │   │               │   │   ├── cp_parser_late_parse_one_default_arg(cp_parser *, tree, tree, tree)
    │       │           │       │   │               │   │   └── pop_unparsed_function_queues(cp_parser *)
    │       │           │       │   │               │   ├── inject_parm_decls(tree)
    │       │           │       │   │               │   ├── cp_parser_late_noexcept_specifier(cp_parser *, tree)/
    │       │           │       │   │               │   │   └── cp_parser_noexcept_specification_opt(cp_parser *, cp_parser_flags, bool, bool *, bool)
    │       │           │       │   │               │   ├── noexcept_override_late_checks(tree, tree)/
    │       │           │       │   │               │   │   └── tree_not_check2(tree, const char *, int, const char *, enum tree_code, enum tree_code)
    │       │           │       │   │               │   ├── pop_injected_parms()
    │       │           │       │   │               │   ├── inject_this_parameter(tree, cp_cv_quals)
    │       │           │       │   │               │   ├── cp_parser_late_parsing_nsdmi(cp_parser *, tree)
    │       │           │       │   │               │   ├── cp_parser_late_contract_condition(cp_parser *, tree, tree)/
    │       │           │       │   │               │   │   └── cp_parser_conditional_expression(cp_parser *)
    │       │           │       │   │               │   └── cp_parser_late_parsing_for_member(cp_parser *, tree)/
    │       │           │       │   │               │       ├── cp_parser_omp_declare_reduction_exprs(tree, cp_parser *)
    │       │           │       │   │               │       └── cp_parser_function_definition_after_declarator(cp_parser *, bool)
    │       │           │       │   │               ├── vec::pop()
    │       │           │       │   │               ├── invoke_plugin_callbacks(int, void *)
    │       │           │       │   │               ├── cp_parser_elaborated_type_specifier(cp_parser *, bool, bool)/
    │       │           │       │   │               │   ├── cp_parser_nested_name_specifier(cp_parser *, bool, bool, bool, bool)
    │       │           │       │   │               │   ├── cp_parser_diagnose_invalid_type_name(cp_parser *, tree, location_t)
    │       │           │       │   │               │   ├── cp_parser_make_typename_type(cp_parser *, tree, location_t)
    │       │           │       │   │               │   ├── cp_parser_declares_only_class_p(cp_parser *)
    │       │           │       │   │               │   └── cp_parser_maybe_warn_enum_key(cp_parser *, location_t, tree, rid)/
    │       │           │       │   │               │       └── cp_parser_lookup_name_simple(cp_parser *, tree, location_t)
    │       │           │       │   │               └── cp_parser_simple_type_specifier(cp_parser *, cp_decl_specifier_seq *, cp_parser_flags)/
    │       │           │       │   │                   ├── cp_parser_abort_tentative_parse(cp_parser *)
    │       │           │       │   │                   ├── synthesize_implicit_template_parm(cp_parser *, tree)/
    │       │           │       │   │                   │   ├── function_being_declared_is_template_p(cp_parser *)
    │       │           │       │   │                   │   ├── tree_vec_elt_check(tree, int, const char *, int, const char *)
    │       │           │       │   │                   │   └── make_generic_type_name()
    │       │           │       │   │                   ├── cp_parser_decltype(cp_parser *)/
    │       │           │       │   │                   │   ├── cp_parser_require_keyword(cp_parser *, enum rid, required_token)
    │       │           │       │   │                   │   ├── cp_parser_commit_to_topmost_tentative_parse(cp_parser *)
    │       │           │       │   │                   │   ├── cp_parser_decltype_expr(cp_parser *, bool &)
    │       │           │       │   │                   │   ├── tree_strip_any_location_wrapper(tree)
    │       │           │       │   │                   │   └── make_location(location_t, location_t, cp_lexer *)
    │       │           │       │   │                   ├── cp_parser_sizeof_operand(cp_parser *, enum rid)/
    │       │           │       │   │                   │   ├── cp_parser_sizeof_pack(cp_parser *)
    │       │           │       │   │                   │   ├── cp_parser_compound_literal_p(cp_parser *)
    │       │           │       │   │                   │   ├── cp_parser_type_id(cp_parser *, cp_parser_flags, location_t *)
    │       │           │       │   │                   │   └── cp_parser_unary_expression(cp_parser *, cp_id_kind *, bool, bool, bool)
    │       │           │       │   │                   ├── cp_parser_trait(cp_parser *, enum rid)
    │       │           │       │   │                   ├── concept_check_p(const_tree)/
    │       │           │       │   │                   │   ├── tree_operand_check(const_tree, int, const char *, int, const char *)
    │       │           │       │   │                   │   └── concept_definition_p(tree)
    │       │           │       │   │                   ├── cp_parser_type_name(cp_parser *, bool)/
    │       │           │       │   │                   │   └── cp_parser_nonclass_name(cp_parser *)
    │       │           │       │   │                   └── cp_parser_placeholder_type_specifier(cp_parser *, location_t, tree, bool)
    │       │           │       │   └── maybe_add_cast_fixit(rich_location *, location_t, location_t, tree, tree)/
    │       │           │       │       └── get_cast_suggestion(tree, tree)
    │       │           │       ├── cp_expr::get_start()
    │       │           │       └── operator==(const cp_expr &, tree)
    │       │           └── cp_parser_parenthesized_expression_list_elt(cp_parser *, bool, bool, bool *)/
    │       │               ├── cp_lexer_set_source_position(cp_lexer *)/
    │       │               │   └── cp_lexer_peek_token(cp_lexer *)
    │       │               ├── cp_parser_braced_list(cp_parser *, bool *)/
    │       │               │   └── cp_parser_initializer_list(cp_parser *, bool *, bool *)/
    │       │               │       ├── cp_parser_array_designator_p(cp_parser *)/
    │       │               │       │   └── cp_parser_skip_to_closing_square_bracket(cp_parser *)/
    │       │               │       │       └── cp_parser_skip_up_to_closing_square_bracket(cp_parser *)
    │       │               │       └── cp_expr_loc_or_input_loc(const_tree)/
    │       │               │           └── cp_expr_loc_or_loc(const_tree, location_t)
    │       │               ├── cp_parser_constant_expression(cp_parser *, int, bool *, bool)/
    │       │               │   └── cp_parser_assignment_expression(cp_parser *, cp_id_kind *, bool, bool)/
    │       │               │       ├── cp_parser_throw_expression(cp_parser *)
    │       │               │       ├── cp_parser_yield_expression(cp_parser *)/
    │       │               │       │   └── cp_parser_braced_list(cp_parser *, bool *)
    │       │               │       ├── cp_parser_assignment_operator_opt(cp_parser *)
    │       │               │       └── cp_parser_initializer_clause(cp_parser *, bool *)
    │       │               └── cp_parser_assignment_expression(cp_parser *, cp_id_kind *, bool, bool)
    │       └── cp_parser_std_attribute_spec_seq(cp_parser *)/
    │           └── cp_parser_std_attribute_spec(cp_parser *)/
    │               ├── contract_attribute_p(const_tree)
    │               ├── cp_parser_contract_attribute_spec(cp_parser *, tree)/
    │               │   ├── cp_parser_contract_mode_opt(cp_parser *, bool)/
    │               │   │   └── cp_parser_contract_role(cp_parser *)
    │               │   ├── cp_token_cache_new(cp_token *, cp_token *)
    │               │   └── contains_error_p(tree)
    │               ├── cpp_type2name(enum cpp_ttype, unsigned char)
    │               └── cp_parser_std_attribute_list(cp_parser *, tree)/
    │                   ├── cp_parser_std_attribute(cp_parser *, tree)/
    │                   │   ├── cp_parser_omp_directive_args(cp_parser *, tree)/
    │                   │   │   └── cp_parser_skip_balanced_tokens(cp_parser *, size_t)
    │                   │   └── cp_parser_omp_sequence_args(cp_parser *, tree)/
    │                   │       └── cp_parser_required_error(cp_parser *, required_token, bool, location_t)
    │                   └── cp_parser_check_std_attribute(location_t, tree, tree)/
    │                       ├── is_attribute_namespace_p(const char *, const_tree)/
    │                       │   └── is_attribute_p(const char *, const_tree)
    │                       ├── lookup_attribute(const char *, const char *, tree)
    │                       └── from_macro_expansion_at(location_t)
    ├── cp_parser_import_declaration(cp_parser *, module_parse, bool)
    └── cp_parser_toplevel_declaration(cp_parser *)/
        └── cp_parser_declaration(cp_parser *, tree)/
            ├── cp_parser_skip_std_attribute_spec_seq(cp_parser *, size_t)
            ├── cp_parser_handle_statement_omp_attributes(cp_parser *, tree)/
            │   └── cp_lexer_alloc()/
            │       └── vec::create(unsigned int)/
            │           ├── vec::reserve_exact(unsigned int)/
            │           │   ├── vec::reserve(unsigned int, bool)
            │           │   └── vec::reserve(unsigned int, bool)
            │           └── vec::reserve_exact(unsigned int)
            ├── cp_lexer_destroy(cp_lexer *)/
            │   └── vec::release()
            ├── cp_parser_linkage_specification(cp_parser *, tree)/
            │   ├── cp_parser_string_literal(cp_parser *, bool, bool, bool)/
            │   │   └── cp_parser_userdef_string_literal(tree)/
            │   │       ├── cp_literal_operator_id(const char *)
            │   │       ├── lookup_literal_operator(tree, vec<tree, va_gc> *)/
            │   │       │   ├── lkp_iterator::operator++()/
            │   │       │   │   ├── ovl_iterator::operator++()
            │   │       │   │   ├── ovl_iterator::pop(tree)
            │   │       │   │   └── ovl_iterator::maybe_push()
            │   │       │   └── ovl_iterator::operator*()
            │   │       ├── vec::truncate(unsigned int)
            │   │       └── make_string_pack(tree)
            │   └── cp_parser_declaration_seq_opt(cp_parser *)
            ├── cp_parser_explicit_specialization(cp_parser *)/
            │   ├── maybe_show_extern_c_location()
            │   └── cp_parser_single_declaration(cp_parser *, vec<deferred_access_check, va_gc> *, bool, bool, bool *)/
            │       ├── cp_parser_perform_template_parameter_access_checks(vec<deferred_access_check, va_gc> *)
            │       └── cp_parser_init_declarator(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, vec<deferred_access_check, va_gc> *, bool, bool, int, bool *, tree *, location_t *, tree *)/
            │           ├── cp_parser_declarator(cp_parser *, cp_parser_declarator_kind, cp_parser_flags, int *, bool *, bool, bool, bool)/
            │           │   ├── cp_parser_ptr_operator(cp_parser *, tree *, cp_cv_quals *, tree *)/
            │           │   │   └── cp_parser_cv_qualifier_seq_opt(cp_parser *)
            │           │   ├── cp_parser_make_indirect_declarator(enum tree_code, tree, cp_cv_quals, cp_declarator *, tree)/
            │           │   │   ├── make_pointer_declarator(cp_cv_quals, cp_declarator *, tree)
            │           │   │   ├── make_ptrmem_declarator(cp_cv_quals, tree, cp_declarator *, tree)
            │           │   │   └── make_reference_declarator(cp_cv_quals, cp_declarator *, bool, tree)
            │           │   └── cp_parser_direct_declarator(cp_parser *, cp_parser_declarator_kind, cp_parser_flags, int *, bool, bool, bool)/
            │           │       ├── cp_parser_parameter_declaration_clause(cp_parser *, cp_parser_flags)/
            │           │       │   └── cp_parser_parameter_declaration_list(cp_parser *, cp_parser_flags, auto_vec<tree> *)/
            │           │       │       └── cp_parser_parameter_declaration(cp_parser *, cp_parser_flags, bool, bool *)/
            │           │       │           ├── cp_parser_cache_defarg(cp_parser *, bool)/
            │           │       │           │   ├── cp_parser_cache_group(cp_parser *, enum cpp_ttype, unsigned int)
            │           │       │           │   └── cp_parser_parameter_declaration_list(cp_parser *, cp_parser_flags, auto_vec<tree> *)
            │           │       │           ├── declares_constrained_type_template_parameter(tree)/
            │           │       │           │   └── is_constrained_parameter(tree)
            │           │       │           ├── cp_parser_default_type_template_argument(cp_parser *)
            │           │       │           ├── declares_constrained_template_template_parameter(tree)
            │           │       │           ├── cp_parser_default_template_template_argument(cp_parser *)
            │           │       │           └── cp_parser_default_argument(cp_parser *, bool)/
            │           │       │               └── cp_parser_initializer(cp_parser *, bool *, bool *, bool)
            │           │       ├── cp_parser_ref_qualifier_opt(cp_parser *)
            │           │       ├── cp_parser_tx_qualifier_opt(cp_parser *)
            │           │       ├── inject_this_parameter(tree, cp_cv_quals)
            │           │       ├── cp_parser_exception_specification_opt(cp_parser *, cp_parser_flags)/
            │           │       │   ├── cp_parser_noexcept_specification_opt(cp_parser *, cp_parser_flags, bool, bool *, bool)
            │           │       │   └── cp_parser_type_id_list(cp_parser *)
            │           │       ├── cp_parser_late_return_type_opt(cp_parser *, cp_declarator *, tree &)/
            │           │       │   ├── cp_parser_trailing_type_id(cp_parser *)
            │           │       │   ├── cp_parser_requires_clause_opt(cp_parser *, bool)/
            │           │       │   │   ├── cp_parser_constraint_expression(cp_parser *)/
            │           │       │   │   │   └── cp_parser_binary_expression(cp_parser *, bool, bool, enum cp_parser_prec, cp_id_kind *)
            │           │       │   │   └── cp_parser_requires_clause_expression(cp_parser *, bool)/
            │           │       │   │       └── cp_parser_constraint_logical_or_expression(cp_parser *, bool)/
            │           │       │   │           └── cp_parser_constraint_logical_and_expression(cp_parser *, bool)/
            │           │       │   │               └── cp_parser_constraint_primary_expression(cp_parser *, bool)/
            │           │       │   │                   ├── cp_parser_unary_constraint_requires_parens(cp_parser *)
            │           │       │   │                   ├── cp_parser_diagnose_ungrouped_constraint_rich(location_t)
            │           │       │   │                   ├── cp_parser_primary_expression(cp_parser *, bool, bool, bool, cp_id_kind *)/
            │           │       │   │                   │   └── cp_parser_primary_expression(cp_parser *, bool, bool, bool, bool, cp_id_kind *)
            │           │       │   │                   ├── cp_parser_constraint_requires_parens(cp_parser *, bool)
            │           │       │   │                   └── cp_parser_diagnose_ungrouped_constraint_plain(location_t)
            │           │       │   ├── cp_parser_late_parsing_omp_declare_simd(cp_parser *, tree)/
            │           │       │   │   ├── cp_parser_omp_all_clauses(cp_parser *, omp_clause_mask, const char *, cp_token *, bool, int)/
            │           │       │   │   │   ├── cp_parser_omp_clause_name(cp_parser *)
            │           │       │   │   │   ├── cp_parser_omp_clause_bind(cp_parser *, tree, location_t)/
            │           │       │   │   │   │   └── omp_clause_subcode_check(tree, enum omp_clause_code, const char *, int, const char *)
            │           │       │   │   │   ├── cp_parser_omp_clause_collapse(cp_parser *, tree, location_t)/
            │           │       │   │   │   │   ├── check_no_duplicate_clause(tree, enum omp_clause_code, const char *, location_t)
            │           │       │   │   │   │   └── omp_clause_elt_check(tree, int, const char *, int, const char *)
            │           │       │   │   │   ├── cp_parser_omp_var_list(cp_parser *, enum omp_clause_code, tree, bool)/
            │           │       │   │   │   │   └── cp_parser_omp_var_list_no_open(cp_parser *, enum omp_clause_code, tree, bool *, bool)/
            │           │       │   │   │   │       ├── cp_parser_postfix_dot_deref_expression(cp_parser *, enum cpp_ttype, cp_expr, bool, cp_id_kind *, location_t)/
            │           │       │   │   │   │       │   ├── cp_parser_dot_deref_incomplete(tree *, cp_expr *, bool *)
            │           │       │   │   │   │       │   └── cp_parser_pseudo_destructor_name(cp_parser *, tree, tree *, tree *)/
            │           │       │   │   │   │       │       └── cp_parser_template_id(cp_parser *, bool, bool, enum tag_types, bool)
            │           │       │   │   │   │       ├── vec::truncate(unsigned int)
            │           │       │   │   │   │       ├── cp_parser_commit_to_tentative_parse(cp_parser *)
            │           │       │   │   │   │       ├── cp_parser_parse_definitely(cp_parser *)
            │           │       │   │   │   │       └── omp_clause_range_check(tree, enum omp_clause_code, enum omp_clause_code, const char *, int, const char *)
            │           │       │   │   │   ├── cp_parser_omp_clause_default(cp_parser *, tree, location_t, bool)
            │           │       │   │   │   ├── cp_parser_omp_clause_filter(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_final(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_grainsize(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_hint(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_defaultmap(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_if(cp_parser *, tree, location_t, bool)
            │           │       │   │   │   ├── cp_parser_omp_clause_reduction(cp_parser *, enum omp_clause_code, bool, tree)
            │           │       │   │   │   ├── cp_parser_omp_clause_lastprivate(cp_parser *, tree)
            │           │       │   │   │   ├── cp_parser_omp_clause_mergeable(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_nowait(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_num_tasks(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_num_threads(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_order(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_ordered(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_priority(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_schedule(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_untied(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_branch(cp_parser *, enum omp_clause_code, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_cancelkind(cp_parser *, enum omp_clause_code, tree, location_t)
            │           │       │   │   │   ├── wide_int_bitmask::operator&(wide_int_bitmask)
            │           │       │   │   │   ├── wide_int_bitmask::operator<<(int)
            │           │       │   │   │   ├── cp_parser_omp_clause_num_teams(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_thread_limit(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_aligned(cp_parser *, tree)
            │           │       │   │   │   ├── cp_parser_omp_clause_allocate(cp_parser *, tree)
            │           │       │   │   │   ├── wide_int_bitmask::operator>>(int)
            │           │       │   │   │   ├── cp_parser_omp_clause_linear(cp_parser *, tree, bool)
            │           │       │   │   │   ├── cp_parser_omp_clause_affinity(cp_parser *, tree)/
            │           │       │   │   │   │   └── cp_parser_omp_iterators(cp_parser *)
            │           │       │   │   │   ├── cp_parser_omp_clause_depend(cp_parser *, tree, location_t)/
            │           │       │   │   │   │   └── cp_parser_omp_clause_doacross_sink(cp_parser *, location_t, tree, bool)
            │           │       │   │   │   ├── cp_parser_omp_clause_doacross(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_detach(cp_parser *, tree)
            │           │       │   │   │   ├── cp_parser_omp_clause_map(cp_parser *, tree)
            │           │       │   │   │   ├── cp_parser_omp_clause_device(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_dist_schedule(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_proc_bind(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_device_type(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_safelen(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_simdlen(cp_parser *, tree, location_t)
            │           │       │   │   │   ├── cp_parser_omp_clause_nogroup(cp_parser *, tree, location_t)
            │           │       │   │   │   └── cp_parser_omp_clause_orderedkind(cp_parser *, enum omp_clause_code, tree, location_t)
            │           │       │   │   ├── wide_int_bitmask::operator|(wide_int_bitmask)
            │           │       │   │   └── cp_finish_omp_declare_variant(cp_parser *, cp_token *, tree)/
            │           │       │   │       └── cp_parser_omp_context_selector_specification(cp_parser *, bool)/
            │           │       │   │           └── cp_parser_omp_context_selector(cp_parser *, tree, bool)
            │           │       │   └── cp_parser_late_parsing_oacc_routine(cp_parser *, tree)/
            │           │       │       └── cp_parser_oacc_all_clauses(cp_parser *, omp_clause_mask, const char *, cp_token *, bool)/
            │           │       │           ├── cp_parser_oacc_clause_async(cp_parser *, tree)
            │           │       │           ├── cp_parser_oacc_simple_clause(location_t, enum omp_clause_code, tree)
            │           │       │           ├── cp_parser_oacc_data_clause(cp_parser *, pragma_omp_clause, tree)
            │           │       │           ├── cp_parser_oacc_data_clause_deviceptr(cp_parser *, tree)
            │           │       │           ├── cp_parser_oacc_shape_clause(cp_parser *, location_t, omp_clause_code, const char *, tree)
            │           │       │           ├── cp_parser_oacc_single_int_clause(cp_parser *, omp_clause_code, const char *, tree)
            │           │       │           ├── cp_parser_oacc_clause_tile(cp_parser *, location_t, tree)
            │           │       │           └── cp_parser_oacc_clause_wait(cp_parser *, tree)/
            │           │       │               └── cp_parser_oacc_wait_list(cp_parser *, location_t, tree)
            │           │       ├── cp_parser_virt_specifier_seq_opt(cp_parser *)
            │           │       ├── make_call_declarator(cp_declarator *, tree, cp_cv_quals, cp_virt_specifiers, cp_ref_qualifier, tree, tree, tree, tree, tree, location_t)
            │           │       ├── make_array_declarator(cp_declarator *, tree)
            │           │       └── cp_parser_declarator_id(cp_parser *, bool)
            │           ├── cp_parser_check_declarator_template_parameters(cp_parser *, cp_declarator *, location_t)/
            │           │   └── cp_parser_check_declarator_template_parameters(cp_parser *, cp_declarator *, location_t)
            │           ├── warn_about_ambiguous_parse(const cp_decl_specifier_seq *, const cp_declarator *)/
            │           │   ├── get_unqualified_id(cp_declarator *)
            │           │   └── gnu_vector_type_p(const_tree)/
            │           │       ├── tree_class_check(const_tree, const enum tree_code_class, const char *, int, const char *)
            │           │       └── tree_class_check(const_tree, const enum tree_code_class, const char *, int, const char *)
            │           ├── cp_parser_function_definition_from_specifiers_and_declarator(cp_parser *, cp_decl_specifier_seq *, tree, const cp_declarator *)/
            │           │   └── cp_parser_function_definition_after_declarator(cp_parser *, bool)
            │           ├── cp_parser_perform_template_parameter_access_checks(vec<deferred_access_check, va_gc> *)
            │           └── strip_declarator_types(tree, cp_declarator *)
            ├── cp_parser_template_declaration(cp_parser *, bool)
            ├── cp_parser_explicit_instantiation(cp_parser *)/
            │   ├── cp_parser_storage_class_specifier_opt(cp_parser *)
            │   ├── cp_parser_function_specifier_opt(cp_parser *, cp_decl_specifier_seq *)
            │   └── cp_parser_consume_semicolon_at_end_of_statement(cp_parser *)
            └── cp_parser_module_export(cp_parser *)/
                ├── cp_parser_declaration_seq_opt(cp_parser *)
                └── cp_parser_declaration(cp_parser *, tree)/
                    ├── cp_parser_module_declaration(cp_parser *, module_parse, bool)
                    ├── cp_parser_import_declaration(cp_parser *, module_parse, bool)
                    ├── cp_parser_namespace_definition(cp_parser *)/
                    │   └── cp_parser_namespace_body(cp_parser *)
                    ├── cp_parser_objc_declaration(cp_parser *, tree)/
                    │   ├── cp_parser_objc_alias_declaration(cp_parser *)
                    │   ├── cp_parser_objc_class_declaration(cp_parser *)
                    │   ├── cp_parser_objc_protocol_declaration(cp_parser *, tree)/
                    │   │   ├── cp_parser_objc_protocol_refs_opt(cp_parser *)/
                    │   │   │   └── cp_parser_objc_identifier_list(cp_parser *)
                    │   │   └── cp_parser_objc_method_prototype_list(cp_parser *)/
                    │   │       ├── cp_parser_objc_method_signature(cp_parser *, tree *)/
                    │   │       │   ├── cp_parser_objc_method_type(cp_parser *)
                    │   │       │   ├── cp_parser_objc_typename(cp_parser *)/
                    │   │       │   │   └── cp_parser_objc_protocol_qualifiers(cp_parser *)
                    │   │       │   ├── cp_parser_objc_method_keyword_params(cp_parser *, tree *)/
                    │   │       │   │   ├── cp_parser_objc_selector(cp_parser *)
                    │   │       │   │   └── cp_parser_attributes_opt(cp_parser *)
                    │   │       │   └── cp_parser_objc_method_tail_params_opt(cp_parser *, bool *, tree *)/
                    │   │       │       └── cp_parser_parameter_declaration(cp_parser *, cp_parser_flags, bool, bool *)
                    │   │       ├── cp_parser_objc_at_property_declaration(cp_parser *)/
                    │   │       │   └── cp_parser_objc_struct_declaration(cp_parser *)/
                    │   │       │       └── decl_spec_seq_has_spec_p(const cp_decl_specifier_seq *, cp_decl_spec)
                    │   │       ├── cp_parser_objc_method_maybe_bad_prefix_attributes(cp_parser *)
                    │   │       └── cp_parser_objc_interstitial_code(cp_parser *)/
                    │   │           ├── cp_parser_linkage_specification(cp_parser *, tree)
                    │   │           ├── cp_parser_namespace_definition(cp_parser *)
                    │   │           └── cp_parser_block_declaration(cp_parser *, bool)/
                    │   │               ├── cp_parser_asm_definition(cp_parser *)/
                    │   │               │   ├── cp_parser_asm_operand_list(cp_parser *)
                    │   │               │   ├── cp_parser_asm_clobber_list(cp_parser *)
                    │   │               │   ├── cp_parser_asm_label_list(cp_parser *)
                    │   │               │   └── symbol_table::finalize_toplevel_asm(tree)
                    │   │               ├── cp_parser_namespace_alias_definition(cp_parser *)/
                    │   │               │   └── cp_parser_qualified_namespace_specifier(cp_parser *)/
                    │   │               │       └── cp_parser_namespace_name(cp_parser *)
                    │   │               ├── cp_parser_using_directive(cp_parser *)
                    │   │               ├── cp_parser_using_enum(cp_parser *)/
                    │   │               │   ├── cp_parser_simple_type_specifier(cp_parser *, cp_decl_specifier_seq *, cp_parser_flags)
                    │   │               │   ├── make_location(cp_token *, cp_token *, cp_token *)
                    │   │               │   └── finish_using_decl(tree, tree, bool)
                    │   │               ├── cp_parser_alias_declaration(cp_parser *)/
                    │   │               │   └── template_info_decl_check(const_tree, const char *, int, const char *)
                    │   │               ├── cp_parser_using_declaration(cp_parser *, bool)/
                    │   │               │   └── cp_parser_unqualified_id(cp_parser *, bool, bool, bool, bool)/
                    │   │               │       ├── cp_parser_template_id_expr(cp_parser *, bool, bool, bool)
                    │   │               │       ├── cp_parser_operator_function_id(cp_parser *)/
                    │   │               │       │   └── cp_parser_operator(cp_parser *, location_t)/
                    │   │               │       │       └── cp_literal_operator_id(const char *)
                    │   │               │       └── cp_parser_conversion_function_id(cp_parser *)/
                    │   │               │           └── cp_parser_conversion_type_id(cp_parser *)/
                    │   │               │               └── cp_parser_conversion_declarator_opt(cp_parser *)/
                    │   │               │                   └── cp_parser_conversion_declarator_opt(cp_parser *)
                    │   │               ├── cp_parser_static_assert(cp_parser *, bool)
                    │   │               └── cp_parser_simple_declaration(cp_parser *, bool, tree *)/
                    │   │                   ├── cp_parser_decl_specifier_seq(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, int *)/
                    │   │                   │   ├── cp_parser_function_specifier_opt(cp_parser *, cp_decl_specifier_seq *)
                    │   │                   │   ├── cp_parser_set_storage_class(cp_parser *, cp_decl_specifier_seq *, enum rid, cp_token *)
                    │   │                   │   ├── cp_parser_constructor_declarator_p(cp_parser *, cp_parser_flags, bool)/
                    │   │                   │   │   ├── cp_parser_global_scope_opt(cp_parser *, bool)
                    │   │                   │   │   ├── cp_parser_template_name(cp_parser *, bool, bool, bool, enum tag_types, bool *)/
                    │   │                   │   │   │   ├── cp_parser_operator_function_id(cp_parser *)
                    │   │                   │   │   │   └── lookup_attribute(const char *, tree)
                    │   │                   │   │   └── cp_lexer_next_token_is_decl_specifier_keyword(cp_lexer *)
                    │   │                   │   └── cp_parser_type_specifier(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, bool, int *, bool *)
                    │   │                   ├── cp_parser_maybe_commit_to_declaration(cp_parser *, cp_decl_specifier_seq *)/
                    │   │                   │   └── is_constrained_auto(const_tree)
                    │   │                   ├── cp_parser_decomposition_declaration(cp_parser *, cp_decl_specifier_seq *, tree *, location_t *)
                    │   │                   └── cp_parser_init_declarator(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, vec<deferred_access_check, va_gc> *, bool, bool, int, bool *, tree *, location_t *, tree *)
                    │   ├── cp_parser_objc_class_interface(cp_parser *, tree)/
                    │   │   ├── cp_parser_objc_superclass_or_category(cp_parser *, bool, tree *, tree *, bool *)
                    │   │   └── cp_parser_objc_class_ivars(cp_parser *)/
                    │   │       └── cp_parser_objc_visibility_spec(cp_parser *)
                    │   ├── cp_parser_objc_class_implementation(cp_parser *)/
                    │   │   └── cp_parser_objc_method_definition_list(cp_parser *)/
                    │   │       ├── cp_parser_objc_at_synthesize_declaration(cp_parser *)
                    │   │       └── cp_parser_objc_at_dynamic_declaration(cp_parser *)
                    │   └── cp_parser_objc_end_implementation(cp_parser *)
                    ├── cp_parser_objc_valid_prefix_attributes(cp_parser *, tree *)
                    └── cp_parser_block_declaration(cp_parser *, bool)
```




