// src/profile.rs

// -----------------------------------------------------------------------------
// [CASE 1] Profile Feature가 켜져 있을 때 (실제 구현)
// -----------------------------------------------------------------------------
#[cfg(feature = "profile")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

#[cfg(feature = "profile")]
static PROFILER: OnceLock<Mutex<Profiler>> = OnceLock::new();

#[cfg(feature = "profile")]
struct ProfileStats {
    count: u64,
    total: Duration,
    max: Duration,
}

#[cfg(feature = "profile")]
pub struct Profiler {
    data: HashMap<&'static str, ProfileStats>,
}

#[cfg(feature = "profile")]
impl Profiler {
    fn get() -> &'static Mutex<Profiler> {
        PROFILER.get_or_init(|| {
            Mutex::new(Profiler {
                data: HashMap::new(),
            })
        })
    }

    pub fn record(name: &'static str, duration: Duration) {
        let mut p = Self::get().lock().unwrap();
        let entry = p.data.entry(name).or_insert(ProfileStats {
            count: 0,
            total: Duration::ZERO,
            max: Duration::ZERO,
        });
        entry.count += 1;
        entry.total += duration;
        if duration > entry.max {
            entry.max = duration;
        }
    }

    pub fn print_stats() {
        let p = Self::get().lock().unwrap();
        let mut entries: Vec<_> = p.data.iter().collect();
        // 총 소요 시간 내림차순 정렬
        entries.sort_by_key(|e| std::cmp::Reverse(e.1.total));

        println!(
            "\n┌─────────────────────────────┬──────────┬──────────────┬──────────────┬──────────────┐"
        );
        println!(
            "│ {:<27} │ {:>8} │ {:>12} │ {:>12} │ {:>12} │",
            "Scope Name", "Calls", "Total(ms)", "Avg(ms)", "Max(ms)"
        );
        println!(
            "├─────────────────────────────┼──────────┼──────────────┼──────────────┼──────────────┤"
        );
        for (name, stats) in entries {
            let total_ms = stats.total.as_secs_f64() * 1000.0;
            let avg_ms = total_ms / stats.count as f64;
            let max_ms = stats.max.as_secs_f64() * 1000.0;
            println!(
                "│ {:<27} │ {:>8} │ {:>12.2} │ {:>12.2} │ {:>12.2} │",
                name, stats.count, total_ms, avg_ms, max_ms
            );
        }
        println!(
            "└─────────────────────────────┴──────────┴──────────────┴──────────────┴──────────────┘"
        );
    }
}

#[cfg(feature = "profile")]
pub struct Scope {
    name: &'static str,
    start: Instant,
}

#[cfg(feature = "profile")]
impl Scope {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

#[cfg(feature = "profile")]
impl Drop for Scope {
    fn drop(&mut self) {
        Profiler::record(self.name, self.start.elapsed());
    }
}

// 매크로 (ON 버전): Scope를 생성하여 시간을 측정합니다.
#[macro_export]
#[cfg(feature = "profile")]
macro_rules! profile {
    ($name:expr) => {
        let _scope = crate::profile::Scope::new($name);
    };
}

// -----------------------------------------------------------------------------
// [CASE 2] Profile Feature가 꺼져 있을 때 (Zero-Cost Dummy)
// -----------------------------------------------------------------------------

#[cfg(not(feature = "profile"))]
pub struct Profiler;

#[cfg(not(feature = "profile"))]
impl Profiler {
    // 빈 함수: 컴파일러가 호출 자체를 제거(Optimize Out)합니다.
    pub fn print_stats() {}
}

// 매크로 (OFF 버전): 아무 코드도 생성하지 않습니다 (No-Op).
#[macro_export]
#[cfg(not(feature = "profile"))]
macro_rules! profile {
    ($name:expr) => {
        // 컴파일 시 완전히 사라집니다.
    };
}
