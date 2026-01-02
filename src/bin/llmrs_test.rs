fn main() {
    // 로거 초기화 (필요시)
    println!("Starting Mobile Test Runner...");

    println!("Executing CPU tests...");
    llmrs::test::cpu::run_all_tests();

    println!("Executing OpenCL tests...");
    llmrs::test::opencl::run_all_tests();

    println!("All tests completed.");
}
