import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = (children: ReactNode, props: IconProps) => (
  <svg
    aria-hidden="true"
    fill="none"
    height="20"
    viewBox="0 0 24 24"
    width="20"
    {...props}
  >
    {children}
  </svg>
);

export const GridIcon = (props: IconProps) =>
  base(
    <>
      <rect height="7" rx="2" stroke="currentColor" strokeWidth="1.8" width="7" x="3" y="3" />
      <rect height="7" rx="2" stroke="currentColor" strokeWidth="1.8" width="7" x="14" y="3" />
      <rect height="7" rx="2" stroke="currentColor" strokeWidth="1.8" width="7" x="3" y="14" />
      <rect height="7" rx="2" stroke="currentColor" strokeWidth="1.8" width="7" x="14" y="14" />
    </>,
    props,
  );

export const UploadIcon = (props: IconProps) =>
  base(
    <>
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
      <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </>,
    props,
  );

export const SlidersIcon = (props: IconProps) =>
  base(
    <>
      <path d="M4 7h10M18 7h2M4 17h2M10 17h10" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
      <circle cx="16" cy="7" r="2" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="8" cy="17" r="2" stroke="currentColor" strokeWidth="1.8" />
    </>,
    props,
  );

export const PlayIcon = (props: IconProps) =>
  base(<path d="m8 5 11 7-11 7V5Z" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />, props);

export const ChartIcon = (props: IconProps) =>
  base(
    <>
      <path d="M4 20V10m6 10V4m6 16v-7m4 7H2" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </>,
    props,
  );

export const ClockIcon = (props: IconProps) =>
  base(
    <>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 7v5l3 2" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </>,
    props,
  );

export const CheckIcon = (props: IconProps) =>
  base(<path d="m5 12 4.2 4L19 7" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />, props);

export const ArrowIcon = (props: IconProps) =>
  base(<path d="m9 5 7 7-7 7" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />, props);

export const DatabaseIcon = (props: IconProps) =>
  base(
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" stroke="currentColor" strokeWidth="1.8" />
      <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" stroke="currentColor" strokeWidth="1.8" />
    </>,
    props,
  );
