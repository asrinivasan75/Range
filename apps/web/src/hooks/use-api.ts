"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface UseApiOptions<T> {
  initialData?: T;
  pollInterval?: number;
  enabled?: boolean;
}

interface UseApiResult<T> {
  data: T | undefined;
  error: Error | null;
  loading: boolean;
  refetch: () => Promise<void>;
}

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  options: UseApiOptions<T> = {}
): UseApiResult<T> {
  const { initialData, pollInterval, enabled = true } = options;
  const [data, setData] = useState<T | undefined>(initialData);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    if (!enabled) return;
    try {
      setLoading(true);
      const result = await fetcher();
      if (mountedRef.current) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e : new Error(String(e)));
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [fetcher, enabled]);

  useEffect(() => {
    mountedRef.current = true;
    fetchData();
    return () => {
      mountedRef.current = false;
    };
  }, [...deps, enabled]);

  useEffect(() => {
    if (!pollInterval || !enabled) return;
    const interval = setInterval(fetchData, pollInterval);
    return () => clearInterval(interval);
  }, [pollInterval, enabled, fetchData]);

  return { data, error, loading, refetch: fetchData };
}
